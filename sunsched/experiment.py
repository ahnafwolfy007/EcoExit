"""Build and run simulated deployments; used by every script under pipeline/.

One job is one policy, on one camera, in one weather year, at one regime cell
(battery autonomy x harvest-to-demand ratio), optionally with config overrides
for sweeps. Jobs are independent, so they run in parallel across processes,
and each worker loads the shared data once.
"""
import copy
import dataclasses
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from sunsched import config as C
from sunsched.env.events import build_streams
from sunsched.env.forecast import ReserveForecaster
from sunsched.env.solar import (battery_temperature, dawn_dusk, harvest_j,
                                irradiation_j_per_m2, load_year)
from sunsched.eval.metrics import summarize
from sunsched.policies.base import RunContext
from sunsched.policies.baselines import (AlwaysNow, ChargeCeilingEarlyExit,
                                         EnergyAwareEarlyExit, LazyAnimalDeferral,
                                         LazyDeferral, TriageOnly)
from sunsched.policies.ours import SunSched
from sunsched.sim.node import build_node_energy
from sunsched.sim.outcomes import load_outcomes
from sunsched.sim.simulator import simulate

BASELINES = ["always_now", "triage_only", "ee_now", "ceiling_now", "lazy_defer", "lazy_animal"]
OURS = ["sunsched"]
ABLATIONS = ["sunsched_no_gate", "sunsched_no_defer", "sunsched_no_ceiling", "sunsched_no_conformal"]


# -- config (de)serialisation, so worker processes rebuild the exact config ----
def config_to_dict(cfg: C.Config) -> dict:
    return dataclasses.asdict(cfg)


def config_from_dict(d: dict) -> C.Config:
    kwargs = {}
    for f in dataclasses.fields(C.Config):
        if f.name not in d:
            continue
        value = d[f.name]
        cls = f.type if not isinstance(f.type, str) else getattr(C, f.type, None)
        kwargs[f.name] = cls(**value) if dataclasses.is_dataclass(cls) and isinstance(value, dict) else value
    return C.Config(**kwargs)


def with_overrides(cfg: C.Config, overrides: Optional[Dict[str, object]]) -> C.Config:
    """Apply {"section.field": value} overrides to a copy."""
    cfg = copy.deepcopy(cfg)
    for key, value in (overrides or {}).items():
        section, field = key.split(".", 1)
        obj = getattr(cfg, section)
        if not hasattr(obj, field):
            raise KeyError(f"unknown config field {key}")
        setattr(obj, field, value)
    return cfg


# -- policies ------------------------------------------------------------------
def make_policy(name: str, cfg: C.Config):
    b, c = cfg.baselines, cfg.control
    interval = max(int(round(c.batch_interval_min * 60 / cfg.slot_seconds)), 1)
    table = {
        "always_now": lambda: AlwaysNow(),
        "triage_only": lambda: TriageOnly(),
        "ee_now": lambda: EnergyAwareEarlyExit(b.ee_theta_hi, b.ee_soc_hi, b.ee_soc_lo),
        "ceiling_now": lambda: ChargeCeilingEarlyExit(b.ee_theta_hi, b.ee_soc_hi, b.ee_soc_lo,
                                                      b.ceiling_margin, c.min_ceiling),
        "lazy_defer": lambda: LazyDeferral(b.lazy_theta, b.lazy_soc_on, b.lazy_soc_keep, interval),
        "lazy_animal": lambda: LazyAnimalDeferral(b.lazy_animal_theta, b.lazy_soc_on,
                                                  b.lazy_soc_keep, interval),
        "sunsched": lambda: SunSched(c),
        "sunsched_no_gate": lambda: SunSched(c, gate=False, name="sunsched_no_gate"),
        "sunsched_no_defer": lambda: SunSched(c, defer=False, name="sunsched_no_defer"),
        "sunsched_no_ceiling": lambda: SunSched(c, ceiling_on=False, name="sunsched_no_ceiling"),
        "sunsched_no_conformal": lambda: SunSched(c, conformal=False, name="sunsched_no_conformal"),
    }
    if name not in table:
        raise ValueError(f"unknown policy {name}")
    return table[name]()


# -- shared data, loaded once per process --------------------------------------
_WORLD: Dict[tuple, dict] = {}


def load_world(cfg: C.Config) -> dict:
    key = (os.path.abspath(cfg.out_dir), cfg.solar.slot_minutes, cfg.experiment.days)
    if key in _WORLD:
        return _WORLD[key]
    art = cfg.artifacts_dir
    outcomes = load_outcomes(f"{art}/outcomes.npz")
    ev = np.load(f"{art}/events.npz")
    streams = build_streams(ev["eval_locations"].astype(str), ev["eval_slot_in_year"],
                            cfg.slots_per_day, cfg.experiment.days)
    world = dict(outcomes=outcomes, streams=streams, solar={}, art=art)
    _WORLD[key] = world
    return world


def solar_year(world: dict, site: str, year: int):
    k = (site, year)
    if k not in world["solar"]:
        path = f"{world['art']}/solar/{site}_{year}.npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing; run pipeline/03_build_environment.py")
        world["solar"][k] = load_year(path)
    return world["solar"][k]


# -- sizing ----------------------------------------------------------------------
def always_now_demand_j(slots: np.ndarray, night: np.ndarray, n_slots: int, energy) -> float:
    """Energy the classify-everything-now policy would need over the window:
    the denominator of the harvest-to-demand ratio."""
    capture = np.where(night[slots], energy.capture_night_j, energy.capture_day_j).sum()
    n_wakes = len(np.unique(slots))
    return float(n_slots * energy.sleep_j + capture + n_wakes * energy.wake_j
                 + len(slots) * energy.refine_j["full"])


def essential_j_per_day(slots: np.ndarray, night: np.ndarray, n_slots: int, energy,
                        days: float) -> float:
    """Mean daily energy of the node's non-negotiable work: sleeping, capturing
    and triaging every frame. Battery autonomy is measured in these days."""
    capture = np.where(night[slots], energy.capture_night_j, energy.capture_day_j).sum()
    total = n_slots * energy.sleep_j + capture + len(slots) * (energy.triage_j + energy.store_j)
    return float(total / max(days, 1e-9))


def panel_area_m2(ratio: float, demand_j: float, site: str, years, window: slice,
                  world: dict, cfg: C.Config) -> float:
    """One panel per (camera, site), sized on the mean over weather years so that
    each year is a genuine weather replicate for the same hardware."""
    per_m2 = np.mean([irradiation_j_per_m2(solar_year(world, site, y).irradiance[window],
                                           cfg.solar, cfg.slot_seconds) for y in years])
    return ratio * demand_j / max(per_m2, 1e-9)


# -- one job ---------------------------------------------------------------------
def run_job(job: dict) -> dict:
    cfg = with_overrides(config_from_dict(job["cfg"]), job.get("overrides"))
    world = load_world(cfg)
    outcomes = world["outcomes"]
    stream = world["streams"][job["location"]]
    site, year, ratio = job["site"], int(job["year"]), float(job["ratio"])
    autonomy = job.get("autonomy")

    spd = cfg.slots_per_day
    n = stream.n_days * spd
    window = slice(stream.day_offset * spd, stream.day_offset * spd + n)
    sy = solar_year(world, site, year)
    irr = sy.irradiance[window]
    elev = sy.elevation_deg[window]
    night = elev < 0.0

    energy = build_node_energy(cfg.node, outcomes.macs, cfg.slot_seconds)
    essential = essential_j_per_day(stream.slot, night, n, energy, stream.n_days)
    if autonomy is not None:
        cfg.battery.capacity_wh = max(float(autonomy) * essential, 1.0) / 3600.0
    demand = always_now_demand_j(stream.slot, night, n, energy)
    area = panel_area_m2(ratio, demand, site, cfg.solar.years, window, world, cfg)
    harvest = harvest_j(irr, area, cfg.solar, cfg.slot_seconds)
    temp = battery_temperature(sy.air_temp_c[window], irr, cfg.solar.enclosure_gain_c,
                               cfg.solar.thermal_tau_hours, cfg.slot_seconds)
    sunrise, sunset = dawn_dusk(elev, spd)

    t0 = time.time()
    soc = None
    for _ in range(max(1, cfg.experiment.steady_state_passes)):
        policy = make_policy(job["policy"], cfg)
        forecaster = ReserveForecaster(
            n_slots=n, horizon_slots=int(round(cfg.control.reserve_horizon_h * 3600 / cfg.slot_seconds)),
            alpha=cfg.control.alpha, gamma=cfg.control.aci_gamma,
            window_days=cfg.control.reserve_window_days,
            eta_charge=cfg.battery.eta_charge, eta_discharge=cfg.battery.eta_discharge,
            conformal=getattr(policy, "uses_conformal", True))
        ctx = RunContext(cfg=cfg, energy=energy, forecaster=forecaster, slots_per_day=spd,
                         slot_seconds=cfg.slot_seconds,
                         deadline_slots=int(round(cfg.control.deadline_h * 3600 / cfg.slot_seconds)),
                         sunrise=sunrise, sunset=sunset, n_slots=n)
        res = simulate(ctx, policy, stream.frame_idx, stream.slot, outcomes, harvest, temp, night,
                       soc_start=soc)
        soc = float(res.soc[-1])
    metrics = summarize(res, outcomes, cfg, stream.n_days)
    return dict(policy=job["policy"], location=job["location"], site=site, year=year,
                ratio=ratio, autonomy=(float(autonomy) if autonomy is not None else float("nan")),
                capacity_wh=cfg.battery.capacity_wh, essential_j_per_day=essential,
                solar_source=sy.source, days=stream.n_days,
                panel_area_cm2=area * 1e4, demand_always_now_kj=demand / 1e3,
                sim_seconds=round(time.time() - t0, 2),
                **{k: v for k, v in job.items() if k.startswith("tag_")}, **metrics)


def _safe_run(job: dict) -> dict:
    try:
        return run_job(job)
    except Exception as e:                  # one bad job should not lose the sweep
        import traceback
        return dict(policy=job.get("policy"), location=job.get("location"), site=job.get("site"),
                    year=job.get("year"), ratio=job.get("ratio"), autonomy=job.get("autonomy"),
                    error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(),
                    **{k: v for k, v in job.items() if k.startswith("tag_")})


def make_jobs(cfg: C.Config, policies: Sequence[str], cells: Sequence[tuple],
              locations=None, years=None, site: str = None,
              overrides: Optional[dict] = None, tags: Optional[dict] = None) -> List[dict]:
    """`cells` is a list of (autonomy_days, ratio); autonomy None keeps the
    battery at BatteryCfg.capacity_wh."""
    world = load_world(cfg)
    locations = locations or sorted(world["streams"])
    years = years or list(cfg.solar.years)
    site = site or cfg.experiment.main_site
    base = config_to_dict(cfg)
    jobs = []
    for p in policies:
        for autonomy, ratio in cells:
            for loc in locations:
                for y in years:
                    job = dict(cfg=base, policy=p, location=loc, year=int(y), ratio=float(ratio),
                               autonomy=autonomy, site=site, overrides=overrides or {})
                    for k, v in (tags or {}).items():
                        job[f"tag_{k}"] = v
                    jobs.append(job)
    return jobs


def grid_cells(cfg: C.Config) -> List[tuple]:
    return [(a, r) for a in cfg.experiment.autonomy_days for r in cfg.experiment.ratios]


def run_jobs(jobs: List[dict], n_workers: int = 0, label: str = "jobs") -> List[dict]:
    if not jobs:
        return []
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    rows = []
    print(f"[{label}] {len(jobs)} simulated deployments on {n_workers} worker(s)", flush=True)
    if n_workers == 1:
        it, pool = map(_safe_run, jobs), None
    else:
        pool = mp.get_context("spawn").Pool(n_workers)
        it = pool.imap_unordered(_safe_run, jobs, chunksize=4)
    try:
        for i, row in enumerate(it, 1):
            rows.append(row)
            if i == len(jobs) or i % max(1, len(jobs) // 20) == 0:
                el = time.time() - t0
                print(f"[{label}]   {i}/{len(jobs)} done, {el / 60:.1f} min elapsed, "
                      f"~{el / i * (len(jobs) - i) / 60:.1f} min left", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"[{label}] WARNING: {len(errors)} job(s) failed; first error:\n{errors[0]['traceback']}")
    return rows
