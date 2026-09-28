"""Build and run simulated deployments; used by every script under pipeline/.

One job = one policy x one camera location x one weather year x one regime,
optionally with config overrides (for sweeps). Jobs are independent, so they
run in parallel across processes; each worker loads the shared data once.
"""
import copy
import dataclasses
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional

import numpy as np

from sunsched import config as C
from sunsched.env.events import build_streams
from sunsched.env.forecast import ReserveForecaster
from sunsched.env.solar import (battery_temperature, dawn_dusk, harvest_j,
                                irradiation_j_per_m2, load_year)
from sunsched.eval.metrics import summarize
from sunsched.policies.base import RunContext
from sunsched.policies.baselines import (AlwaysNow, ChargeCeilingEarlyExit,
                                         EnergyAwareEarlyExit, LazyDeferral, TriageOnly)
from sunsched.policies.ours import SunSched
from sunsched.sim.node import build_node_energy
from sunsched.sim.outcomes import load_outcomes
from sunsched.sim.simulator import simulate

BASELINES = ["always_now", "triage_only", "ee_now", "lazy_defer", "ceiling_now"]
OURS = ["sunsched"]
ABLATIONS = ["sunsched_no_defer", "sunsched_no_ceiling", "sunsched_no_conformal", "sunsched_lite"]
# The per-frame and deferral controllers the kill test compares against.
STRONG_BASELINES = ["ee_now", "ceiling_now", "lazy_defer"]


# -- config (de)serialisation, so worker processes rebuild the exact config ----
def config_to_dict(cfg: C.Config) -> dict:
    return dataclasses.asdict(cfg)


def config_from_dict(d: dict) -> C.Config:
    sections = {f.name: f.type for f in dataclasses.fields(C.Config)}
    kwargs = {}
    for name, value in d.items():
        cls = getattr(C, sections[name], None) if isinstance(sections.get(name), str) else sections.get(name)
        if dataclasses.is_dataclass(cls) and isinstance(value, dict):
            kwargs[name] = cls(**value)
        else:
            kwargs[name] = value
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
    if name == "always_now":
        return AlwaysNow()
    if name == "triage_only":
        return TriageOnly()
    if name == "ee_now":
        return EnergyAwareEarlyExit(b.ee_theta_hi, b.ee_soc_hi, b.ee_soc_lo)
    if name == "lazy_defer":
        return LazyDeferral(b.lazy_theta, b.lazy_soc_on, b.lazy_soc_keep, interval)
    if name == "ceiling_now":
        return ChargeCeilingEarlyExit(b.ee_theta_hi, b.ee_soc_hi, b.ee_soc_lo,
                                      b.ceiling_margin, c.min_ceiling)
    if name == "sunsched":
        return SunSched(c)
    if name == "sunsched_no_defer":
        return SunSched(c, defer=False, name=name)
    if name == "sunsched_no_ceiling":
        return SunSched(c, ceiling_on=False, name=name)
    if name == "sunsched_no_conformal":
        return SunSched(c, conformal=False, name=name)
    if name == "sunsched_lite":
        return SunSched(c, refine_op="lite", name=name)
    raise ValueError(f"unknown policy {name}")


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
    """Energy the run-everything-now policy would need over the window.

    This is the denominator of the harvest-to-demand ratio: above 1 there is
    enough sun to classify every capture with the full model on arrival, and
    scheduling can only help the battery, not the accuracy.
    """
    capture = np.where(night[slots], energy.capture_night_j, energy.capture_day_j).sum()
    n_wakes = len(np.unique(slots))
    return float(n_slots * energy.sleep_j + capture + n_wakes * energy.wake_j
                 + len(slots) * energy.refine_j["full"])


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

    spd = cfg.slots_per_day
    n = stream.n_days * spd
    window = slice(stream.day_offset * spd, stream.day_offset * spd + n)
    sy = solar_year(world, site, year)
    irr = sy.irradiance[window]
    elev = sy.elevation_deg[window]
    night = elev < 0.0

    energy = build_node_energy(cfg.node, outcomes.macs, cfg.slot_seconds)
    demand = always_now_demand_j(stream.slot, night, n, energy)
    # Optional sizing axis. Fixing the battery in days of load rather than in Wh
    # makes "autonomy" mean the same thing at every camera, whose capture rates
    # differ threefold. It happens here because the load is only known once the
    # stream and the node energy model exist. A night's reserve is a real
    # fraction of a few days of store and a rounding error on a few months of
    # it, so this is the axis that decides whether scheduling can matter at all.
    if job.get("autonomy_days"):
        cfg.battery.capacity_wh = float(job["autonomy_days"]) * (demand / stream.n_days) / 3600.0
    area = panel_area_m2(ratio, demand, site, cfg.solar.years, window, world, cfg)
    harvest = harvest_j(irr, area, cfg.solar, cfg.slot_seconds)
    temp = battery_temperature(sy.air_temp_c[window], irr, cfg.solar.enclosure_gain_c,
                               cfg.solar.thermal_tau_hours, cfg.slot_seconds)
    sunrise, sunset = dawn_dusk(elev, spd)

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

    t0 = time.time()
    res = simulate(ctx, policy, stream.frame_idx, stream.slot, outcomes, harvest, temp, night)
    metrics = summarize(res, outcomes, cfg, stream.n_days)
    return dict(policy=job["policy"], location=job["location"], site=site, year=year,
                ratio=ratio, solar_source=sy.source, days=stream.n_days,
                panel_area_cm2=area * 1e4, demand_always_now_kj=demand / 1e3,
                capacity_wh=cfg.battery.capacity_wh,
                days_of_autonomy=cfg.battery.capacity_wh * 3600.0 / max(demand / stream.n_days, 1e-9),
                sim_seconds=round(time.time() - t0, 2),
                **{k: v for k, v in job.items() if k.startswith("tag_")}, **metrics)


def _safe_run(job: dict) -> dict:
    try:
        return run_job(job)
    except Exception as e:                  # one bad job should not lose the sweep
        import traceback
        return dict(policy=job.get("policy"), location=job.get("location"), site=job.get("site"),
                    year=job.get("year"), ratio=job.get("ratio"), error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc(),
                    **{k: v for k, v in job.items() if k.startswith("tag_")})


def make_jobs(cfg: C.Config, policies: List[str], ratios, locations=None, years=None,
              site: str = None, overrides: Optional[dict] = None, tags: Optional[dict] = None,
              autonomy_days: Optional[float] = None) -> List[dict]:
    world = load_world(cfg)
    locations = locations or sorted(world["streams"])
    years = years or list(cfg.solar.years)
    site = site or cfg.experiment.main_site
    base = config_to_dict(cfg)
    jobs = []
    for p in policies:
        for r in ratios:
            for loc in locations:
                for y in years:
                    job = dict(cfg=base, policy=p, location=loc, year=int(y), ratio=float(r),
                               site=site, overrides=overrides or {})
                    if autonomy_days:
                        job["autonomy_days"] = float(autonomy_days)
                    for k, v in (tags or {}).items():
                        job[f"tag_{k}"] = v
                    jobs.append(job)
    return jobs


def run_jobs(jobs: List[dict], n_workers: int = 0, label: str = "jobs") -> List[dict]:
    if not jobs:
        return []
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    rows = []
    print(f"[{label}] {len(jobs)} simulated deployments on {n_workers} worker(s)")
    if n_workers == 1:
        it = map(_safe_run, jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(n_workers)
        it = pool.imap_unordered(_safe_run, jobs, chunksize=1)
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
