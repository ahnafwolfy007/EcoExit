"""Shared experiment harness.

The gate script, the experiment runner and the ablation sweeps all need the
same three things: load the trained backbone into an energy/accuracy table,
build a site at a given panel area and seed, and run one policy through the
simulator and summarize it. Having one implementation of each means the gate
and the headline table cannot silently disagree.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import copy
import numpy as np

from ecoexit.config import SITES
from ecoexit.control.pricing import carbon_kappa
from ecoexit.forecast.conformal import simulate_forecast_and_calibration
from ecoexit.sim.loop import run_shadow_price, run_fixed_policy
from ecoexit.sim.stream import (
    FrameStream, make_stream, make_stream_from_events, value_weighted_recall,
)
from ecoexit.solar.traces import make_trace
from ecoexit.wear.rainflow import CycleLifeCurve, offline_rainflow_wear
from ecoexit.eval.metrics import (
    useful_inferences_per_joule, amortised_embodied_carbon_kg,
    projected_replacements_continuous, carbon_normalised_task_utility,
    gco2e_per_correct_inference,
)


# -- loading -----------------------------------------------------------------
def load_backbone(artifacts_dir: str, cfg, use_val_grid: bool = True) -> Dict:
    """Load the trained model's energy table, accuracy grid and per-image outcomes.

    `use_val_grid=True` hands the controller the *validation* accuracy grid as
    its value function. The original pipeline used the test grid, which meant
    the controller was tuned on the data it was then scored on. The test grid is
    still loaded, under `acc_grid_test`, and used only for reporting.
    """
    import torch
    from ecoexit.models.elastic import ElasticNet, set_resolution_idx
    from ecoexit.energy.model import profile_model, DecomposedModel, build_energy_table

    ckpt = torch.load(f"{artifacts_dir}/backbone.pt", map_location="cpu", weights_only=False)
    model = ElasticNet(ckpt["resolutions"], ckpt["n_exits"], ckpt["base_width"],
                       ckpt["blocks_per_stage"], ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    resolutions = tuple(int(r) for r in ckpt["resolutions"])
    n_exits = int(ckpt["n_exits"])

    points = profile_model(model, resolutions, n_exits, set_resolution_idx, n_timing_runs=30)
    decomposed = DecomposedModel(cfg.energy)
    raw_table = build_energy_table(points, decomposed)

    sf = cfg.energy.system_scale_factor
    energy_table = {k: v * sf for k, v in raw_table.items()}
    sense_energy = {ri: decomposed.sense_energy_j(r) * sf for ri, r in enumerate(resolutions)}

    grids = np.load(f"{artifacts_dir}/final_accuracy_grid.npz")
    acc_test = grids["acc_grid"]
    acc_val = grids["acc_grid_val"] if "acc_grid_val" in grids else acc_test

    outc = np.load(f"{artifacts_dir}/per_image_outcomes.npz")
    correct, conf = outc["correct"], outc["conf"]
    eval_pool = outc["eval_pool"] if "eval_pool" in outc else np.arange(correct.shape[2])

    per_image = {ri: {e: (correct[ri, e], conf[ri, e]) for e in range(n_exits)}
                 for ri in range(len(resolutions))}

    return dict(resolutions=resolutions, n_exits=n_exits, energy_table=energy_table,
                sense_energy=sense_energy,
                acc_grid=(acc_val if use_val_grid else acc_test),
                acc_grid_val=acc_val, acc_grid_test=acc_test,
                per_image=per_image, eval_pool=eval_pool,
                raw_energy_table=raw_table)


# -- site construction -------------------------------------------------------
@dataclass
class SiteData:
    site: str
    harvest_j: np.ndarray
    elevation: np.ndarray
    ghi_clear: np.ndarray
    kc: np.ndarray
    stream: FrameStream
    lower_j_per_day: np.ndarray
    point_j_per_day: np.ndarray
    panel_area_m2: float
    trace_days: float


def build_site(cfg, site: str, seed: int = 0, panel_area_m2: float = None,
               eval_pool_size: int = 10000,
               event_slots: np.ndarray = None) -> SiteData:
    """One site, one seed, at a chosen panel area.

    Panel area is an explicit argument rather than a config constant because
    the regime sweep varies it per site -- see `ecoexit.sizing` for why fixing
    it at one value makes four of five sites uncontrollable.
    """
    solar_cfg = copy.copy(cfg.solar)
    if panel_area_m2 is not None:
        solar_cfg.panel_area_m2 = panel_area_m2

    trace = make_trace(site, SITES[site], solar_cfg, seed=seed,
                       cache_dir=cfg.data.pvgis_cache_dir)
    n = trace.n_slots
    slots_per_day = cfg.slots_per_day

    if event_slots is not None:
        stream = make_stream_from_events(event_slots, n, cfg.stream,
                                         dataset_size=eval_pool_size, seed=seed,
                                         location=site)
    else:
        stream = make_stream(trace.elevation_deg, n, solar_cfg.slot_seconds,
                             cfg.stream, dataset_size=eval_pool_size, seed=seed)

    fc = simulate_forecast_and_calibration(
        trace.kc, slots_per_day, slots_per_day, trace.ghi_clear,
        solar_cfg.panel_area_m2, solar_cfg.panel_efficiency,
        solar_cfg.slot_seconds, cfg.control.alpha, cfg.control.aci_gamma,
    )

    return SiteData(site=site, harvest_j=trace.harvest_j,
                    elevation=trace.elevation_deg, ghi_clear=trace.ghi_clear,
                    kc=trace.kc, stream=stream,
                    lower_j_per_day=fc["lower_j"], point_j_per_day=fc["point_j"],
                    panel_area_m2=solar_cfg.panel_area_m2,
                    trace_days=n / slots_per_day)


# -- running -----------------------------------------------------------------
def run_ours(cfg, bb: Dict, sd: SiteData, wear_mode: str = None,
             confidence_aware: bool = True, use_value_estimate: bool = True,
             lower_bound: np.ndarray = None, kappa_scale: float = None):
    capacity_j = cfg.battery_capacity_j
    kappa = carbon_kappa(cfg.carbon, cfg.battery.capacity_wh)
    kappa *= (cfg.wear.kappa_scale if kappa_scale is None else kappa_scale)
    return run_shadow_price(
        cfg, bb["energy_table"], bb["sense_energy"], bb["acc_grid"], bb["per_image"],
        sd.harvest_j,
        sd.lower_j_per_day if lower_bound is None else lower_bound,
        cfg.slots_per_day, cfg.replan_slots, sd.stream, capacity_j,
        cfg.battery.soc_min_frac * capacity_j, sd.elevation,
        kappa=kappa, wear_mode=(wear_mode or cfg.wear.mode),
        confidence_aware=confidence_aware, use_value_estimate=use_value_estimate,
    )


def run_static(cfg, bb: Dict, sd: SiteData, res_idx: int, exit_idx: int):
    from ecoexit.control.baselines import Action
    capacity_j = cfg.battery_capacity_j
    act = Action(duty=1, res_idx=res_idx, exit_idx=exit_idx)
    return run_fixed_policy(
        cfg, bb["energy_table"], bb["sense_energy"], bb["per_image"], sd.harvest_j,
        sd.stream, capacity_j, cfg.battery.soc_min_frac * capacity_j,
        lambda t, soc: act,
    )


def run_static_max(cfg, bb, sd):
    n_res, n_exit = bb["acc_grid"].shape
    return run_static(cfg, bb, sd, n_res - 1, n_exit - 1)


def run_static_min(cfg, bb, sd):
    return run_static(cfg, bb, sd, 0, 0)


# -- accounting --------------------------------------------------------------
def summarize(policy: str, cfg, result, sd: SiteData, extra: Dict = None) -> dict:
    """One results row, computed identically for every policy."""
    curve = CycleLifeCurve.fit(cfg.battery.dod_points, cfg.battery.cycles_at_dod)
    rf = offline_rainflow_wear(result.soc_frac, curve, min_swing=cfg.wear.min_swing)

    total_value = float((sd.stream.value * result.correct).sum())
    n_correct = int(result.correct.sum())
    total_energy = float(result.energy_j.sum())

    repl = projected_replacements_continuous(rf["wear"], sd.trace_days,
                                             cfg.carbon.deployment_years)
    carbon_kg = amortised_embodied_carbon_kg(cfg.carbon, cfg.battery.capacity_wh, repl)
    vwr = value_weighted_recall(result.captured, sd.stream)

    row = dict(
        policy=policy, site=sd.site, panel_area_m2=sd.panel_area_m2,
        total_value=total_value, n_correct=n_correct,
        total_energy_j=total_energy,
        inferences_per_joule=useful_inferences_per_joule(n_correct, total_energy),
        downtime_frac=float(result.depleted.mean()),
        value_recall=vwr["value_recall"], event_recall=vwr["event_recall"],
        n_events=vwr["n_events"], n_events_captured=vwr["n_events_captured"],
        wear=rf["wear"],
        equiv_full_cycles_per_year=rf["equivalent_full_cycles"] * (365.0 / max(sd.trace_days, 1e-9)),
        projected_replacements=repl,
        carbon_kg=carbon_kg,
        ctu=carbon_normalised_task_utility(total_value, carbon_kg),
        gco2e_per_correct=gco2e_per_correct_inference(carbon_kg, n_correct),
        capture_rate=float(result.captured.mean()),
        controller_overhead_j=float(getattr(result, "controller_overhead_j", 0.0) or 0.0),
    )
    if result.lambda_trace is not None:
        lam = result.lambda_trace
        row.update(lambda_mean=float(lam.mean()), lambda_max=float(lam.max()),
                   lambda_frac_zero=float((lam <= 1e-12).mean()))
    if result.mu_trace is not None:
        mu = result.mu_trace
        row.update(mu_mean=float(mu.mean()), mu_max=float(mu.max()))
    if result.online_wear is not None:
        row["online_wear"] = float(result.online_wear)
        row["wear_online_vs_offline_relerr"] = float(
            abs(result.online_wear - rf["wear"]) / max(rf["wear"], 1e-12))
    if extra:
        row.update(extra)
    return row
