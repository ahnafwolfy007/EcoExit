#!/usr/bin/env python
"""The full closed-loop comparison: every policy x every site, plus the four
headline experiments (value of forecast, risk-accuracy frontier, carbon
ranking inversion, cross-climate transfer) and the knob/confidence ablations.

Usage:
    python scripts/04_run_experiments.py
    python scripts/04_run_experiments.py --skip-rl     # drop the two most
                                                        # expensive baselines
                                                        # (report's "cut first")

Requires scripts/01 and 02 (a trained checkpoint + profiled energy/accuracy)
and scripts/03 (site traces + forecasts) to have already run.

Produces under results/:
    tables/main_comparison.csv       -- every (site, policy) row, all metrics
    tables/value_of_forecast.csv
    tables/risk_frontier.csv
    tables/carbon_inversion.csv
    tables/ablations.csv
    figures/pareto_frontier.png
    figures/vof_sweep.png
    figures/risk_frontier.png
    figures/carbon_inversion.png
    figures/cross_climate_transfer.png
    figures/soc_timeline_example.png
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecoexit.config import Config, TRAIN_SITE, TEST_SITES
from ecoexit.models.elastic import ElasticNet, set_resolution_idx
from ecoexit.energy.model import profile_model, DecomposedModel, build_energy_table
from ecoexit.control.baselines import (
    Action, all_actions, action_energy,
    StaticController, ReactiveLUT, HarvSchedQLearning, MonotoneMDPController, OracleDP,
)
from ecoexit.sim.loop import run_shadow_price, run_fixed_policy, run_action_sequence
from ecoexit.sim.stream import FrameStream, value_weighted_recall
from ecoexit.forecast.conformal import simulate_forecast_and_calibration
from ecoexit.battery.model import Battery
from ecoexit.eval.metrics import (
    useful_inferences_per_joule, total_embodied_carbon_kg, replacements_from_wear,
    carbon_normalised_task_utility, value_of_forecast, percent_of_oracle,
)

PALETTE = dict(ours="#9A6508", reactive="#3B4453", static_max="#9B333E",
               static_min="#C97A83", qlearning="#116460", mdp="#6C7688",
               oracle="#151A23", conf_only="#8A5B12")


# ---------------------------------------------------------------- loading --
def load_backbone(artifacts_dir, cfg):
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

    acc_grid = np.load(f"{artifacts_dir}/final_accuracy_grid.npz")["acc_grid"]
    outc = np.load(f"{artifacts_dir}/per_image_outcomes.npz")
    correct, conf = outc["correct"], outc["conf"]
    per_image = {ri: {e: (correct[ri, e], conf[ri, e]) for e in range(n_exits)}
                 for ri in range(len(resolutions))}
    return dict(resolutions=resolutions, n_exits=n_exits, energy_table=energy_table,
                sense_energy=sense_energy, acc_grid=acc_grid, per_image=per_image)


def load_site(artifacts_dir, site):
    d = np.load(f"{artifacts_dir}/traces/{site}.npz")
    fc = np.load(f"{artifacts_dir}/traces/{site}_forecast.npz")
    stream = FrameStream(value=d["stream_value"], is_event=d["stream_is_event"],
                          cifar_index=d["stream_cifar_index"])
    return dict(ghi=d["ghi"], ghi_clear=d["ghi_clear"], elevation=d["elevation"],
                harvest_j=d["harvest_j"], kc=d["kc"], stream=stream,
                lower_j_per_day=fc["lower_j"], point_j_per_day=fc["point_j"])


# ------------------------------------------------------------- accounting --
def summarize(policy, site, cfg, result, stream, battery, trace_days, oracle_ref=None):
    total_value = float((stream.value * result.correct).sum())
    n_correct = int(result.correct.sum())
    total_energy = float(result.energy_j.sum())
    wear_curve, half_cycles = battery.wear_from_soc_series(result.soc_frac)
    final_wear = float(wear_curve[-1]) if len(wear_curve) else 0.0
    n_repl = replacements_from_wear(final_wear)
    carbon_kg = total_embodied_carbon_kg(cfg.carbon, cfg.battery.capacity_wh, n_repl)
    vwr = value_weighted_recall(result.captured, stream)
    equiv_cycles_yr = 0.5 * sum(d for _, _, d in half_cycles) * (365.0 / max(trace_days, 1e-9))
    overhead = getattr(result, "controller_overhead_j", 0.0) or 0.0

    row = dict(
        policy=policy, site=site,
        value_recall=vwr["value_recall"], event_recall=vwr["event_recall"],
        n_events=vwr["n_events"], n_events_captured=vwr["n_events_captured"],
        total_value=total_value, n_correct=n_correct, total_energy_j=total_energy,
        inferences_per_joule=useful_inferences_per_joule(n_correct, total_energy),
        downtime_frac=float(result.depleted.mean()),
        equiv_full_cycles_per_year=equiv_cycles_yr, final_wear=final_wear,
        n_replacements=n_repl, carbon_kg=carbon_kg,
        ctu=carbon_normalised_task_utility(total_value, carbon_kg),
        controller_overhead_j=overhead,
        pct_of_oracle=np.nan,
    )
    if oracle_ref is not None and oracle_ref > 0:
        row["pct_of_oracle"] = percent_of_oracle(total_value, oracle_ref)
    return row


def action_lists(n_res, n_exit):
    acts = all_actions(n_res, n_exit)
    return acts


# --------------------------------------------------------- baseline fits --
def fit_harvsched(cfg, bb, site_data, n_epochs=3):
    n_res, n_exit = bb["acc_grid"].shape
    acts = action_lists(n_res, n_exit)
    n_actions = len(acts)
    idle_j = cfg.energy.p_idle_w * cfg.solar.slot_seconds
    wake_j = cfg.energy.e_wake_j
    harvest = site_data["harvest_j"]
    stream = site_data["stream"]
    n = len(harvest)
    slots_per_day = int(86400 / cfg.solar.slot_seconds)
    capacity_j = cfg.battery_capacity_j
    soc_min_j = cfg.battery.soc_min_frac * capacity_j

    q = HarvSchedQLearning(cfg.control.soc_bins, n_hour_bins=24, n_actions=n_actions)
    rng = np.random.default_rng(cfg.train.seed)

    for epoch in range(n_epochs):
        soc = capacity_j * cfg.battery.soc_init_frac
        for t in range(n):
            hour_frac = (t % slots_per_day) / slots_per_day
            day_idx = t // slots_per_day
            is_wk = (day_idx % 7) >= 5
            soc_frac = soc / capacity_j
            a_idx = q.act_index(soc_frac, hour_frac, is_wk, rng)
            a = acts[a_idx]
            if a.duty == 0:
                e, correct = idle_j, False
            else:
                img = stream.cifar_index[t]
                correct = bool(bb["per_image"][a.res_idx][a.exit_idx][0][img])
                e = idle_j + wake_j + bb["sense_energy"][a.res_idx] + bb["energy_table"][(a.res_idx, a.exit_idx)]
            frame_val = stream.value[t] * float(correct)
            nxt = soc + cfg.battery.eta_charge * harvest[t] - e / cfg.battery.eta_discharge
            depleted = nxt < soc_min_j
            reward = frame_val - (2.0 if depleted else 0.0)
            soc = float(np.clip(nxt, 0.0, capacity_j))

            nh = ((t + 1) % slots_per_day) / slots_per_day
            nd = (t + 1) // slots_per_day
            nw = (nd % 7) >= 5
            q.update(soc_frac, hour_frac, is_wk, a_idx, reward, soc / capacity_j, nh, nw)
            q.decay_eps()
    return q, acts


def fit_monotone_mdp(cfg, bb, site_data, n_harvest_bins=3):
    n_res, n_exit = bb["acc_grid"].shape
    acts = action_lists(n_res, n_exit)
    idle_j = cfg.energy.p_idle_w * cfg.solar.slot_seconds
    wake_j = cfg.energy.e_wake_j
    ae = np.array([action_energy(a, bb["energy_table"], bb["sense_energy"], idle_j, wake_j) for a in acts])
    base_val = np.array([0.0 if a.duty == 0 else bb["acc_grid"][a.res_idx, a.exit_idx] for a in acts])
    mean_val = float(site_data["stream"].value.mean())
    action_value_const = base_val * mean_val  # Bullo-style: no time/forecast awareness

    harvest = site_data["harvest_j"]
    edges = np.quantile(harvest, [1 / 3, 2 / 3])
    bins = np.digitize(harvest, edges)
    levels = np.array([harvest[bins == b].mean() if (bins == b).any() else harvest.mean()
                        for b in range(n_harvest_bins)])
    trans = np.zeros((n_harvest_bins, n_harvest_bins))
    for b0, b1 in zip(bins[:-1], bins[1:]):
        trans[b0, b1] += 1
    trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1)

    mdp = MonotoneMDPController(cfg.control.soc_bins, n_harvest_bins, len(acts))
    capacity_j = cfg.battery_capacity_j
    soc_min_j = cfg.battery.soc_min_frac * capacity_j
    mdp.solve(ae, action_value_const, capacity_j, soc_min_j, levels, trans, n_iters=150)
    return mdp, acts, edges


# ------------------------------------------------------------- one site ---
def run_all_policies_for_site(cfg, bb, site, site_data, harv_q, harv_acts,
                                mdp, mdp_acts, mdp_edges, lut, skip_rl=False):
    n_res, n_exit = bb["acc_grid"].shape
    idle_j = cfg.energy.p_idle_w * cfg.solar.slot_seconds
    wake_j = cfg.energy.e_wake_j
    capacity_j = cfg.battery_capacity_j
    soc_min_j = cfg.battery.soc_min_frac * capacity_j
    slots_per_day = int(86400 / cfg.solar.slot_seconds)
    replan_slots = int(cfg.control.replan_minutes * 60 / cfg.solar.slot_seconds)

    harvest = site_data["harvest_j"]
    stream = site_data["stream"]
    elevation = site_data["elevation"]
    trace_days = len(harvest) * cfg.solar.slot_seconds / 86400.0

    battery = Battery(cfg.battery, capacity_j).set_slot_seconds(cfg.solar.slot_seconds)
    results = {}

    # -- ours: shadow-price controller ------------------------------------
    t0 = time.time()
    results["ours"] = run_shadow_price(
        cfg, bb["energy_table"], bb["sense_energy"], bb["acc_grid"], bb["per_image"],
        harvest, site_data["lower_j_per_day"], slots_per_day, replan_slots,
        stream, capacity_j, soc_min_j, elevation, confidence_aware=True,
    )
    t_ours = time.time() - t0

    # -- static ceilings / floor -------------------------------------------
    static_max = StaticController(n_res - 1, n_exit - 1)
    results["static_max"] = run_fixed_policy(
        cfg, bb["energy_table"], bb["sense_energy"], bb["per_image"], harvest,
        stream, capacity_j, soc_min_j, lambda t, s: static_max.act())

    static_min = StaticController(0, 0)
    results["static_min"] = run_fixed_policy(
        cfg, bb["energy_table"], bb["sense_energy"], bb["per_image"], harvest,
        stream, capacity_j, soc_min_j, lambda t, s: static_min.act())

    # -- confidence-only early exit (BranchyNet-style), always max res -----
    thresholds = np.full(n_exit - 1, 0.75)
    def conf_only_act(t, s):
        img = stream.cifar_index[t]
        k = 0
        conf = bb["per_image"][n_res - 1][0][1][img]
        while k < n_exit - 1 and conf >= thresholds[k]:
            k += 1
            conf = bb["per_image"][n_res - 1][k][1][img]
        return Action(1, n_res - 1, k)
    results["conf_only"] = run_fixed_policy(
        cfg, bb["energy_table"], bb["sense_energy"], bb["per_image"], harvest,
        stream, capacity_j, soc_min_j, conf_only_act)

    # -- reactive LUT (ePerceptive) -----------------------------------------
    results["reactive"] = run_fixed_policy(
        cfg, bb["energy_table"], bb["sense_energy"], bb["per_image"], harvest,
        stream, capacity_j, soc_min_j, lambda t, s: lut.act(s))

    if not skip_rl:
        # -- HarvSched Q-learning (frozen, fit on TRAIN_SITE only) ---------
        def q_act(t, soc_frac):
            hour_frac = (t % slots_per_day) / slots_per_day
            day_idx = t // slots_per_day
            is_wk = (day_idx % 7) >= 5
            rng_dummy = np.random.default_rng(0)
            idx = harv_q.act_index(soc_frac, hour_frac, is_wk, rng_dummy, greedy=True)
            return harv_acts[idx]
        results["qlearning"] = run_fixed_policy(
            cfg, bb["energy_table"], bb["sense_energy"], bb["per_image"], harvest,
            stream, capacity_j, soc_min_j, q_act)

        # -- Bullo-style monotone MDP (frozen, fit on TRAIN_SITE only) -----
        def mdp_act(t, soc_frac):
            hb = int(np.digitize(harvest[t], mdp_edges))
            idx = mdp.act_index(soc_frac, hb, capacity_j, soc_min_j)
            return mdp_acts[idx]
        results["mdp"] = run_fixed_policy(
            cfg, bb["energy_table"], bb["sense_energy"], bb["per_image"], harvest,
            stream, capacity_j, soc_min_j, mdp_act)

    # -- oracle --------------------------------------------------------------
    acts = action_lists(n_res, n_exit)
    ae = np.array([action_energy(a, bb["energy_table"], bb["sense_energy"], idle_j, wake_j) for a in acts])
    base_val = np.array([0.0 if a.duty == 0 else bb["acc_grid"][a.res_idx, a.exit_idx] for a in acts])
    oracle = OracleDP(cfg.control.oracle_soc_bins)
    oracle_out = oracle.solve(harvest, ae, base_val, capacity_j, soc_min_j,
                                cfg.battery.eta_charge, cfg.battery.eta_discharge,
                                frame_value_t=stream.value)
    results["oracle"] = run_action_sequence(
        cfg, bb["energy_table"], bb["sense_energy"], bb["per_image"], harvest,
        stream, capacity_j, soc_min_j, acts, oracle_out["actions"])

    oracle_value = float((stream.value * results["oracle"].correct).sum())
    rows = [summarize(name, site, cfg, res, stream, battery, trace_days, oracle_value)
            for name, res in results.items()]
    return rows, results, t_ours


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--skip-rl", action="store_true",
                     help="drop HarvSched-QL and Bullo-MDP baselines (faster; report's cut-first item)")
    args = ap.parse_args()

    os.makedirs(f"{args.out}/tables", exist_ok=True)
    os.makedirs(f"{args.out}/figures", exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)

    cfg = Config()
    bb = load_backbone(args.artifacts, cfg)
    print(f"[exp] backbone loaded: resolutions={bb['resolutions']} n_exits={bb['n_exits']}")
    print(f"[exp] energy table (scaled x{cfg.energy.system_scale_factor}): "
          + ", ".join(f"{k}={v:.3f}J" for k, v in sorted(bb["energy_table"].items())))

    all_sites = [TRAIN_SITE] + TEST_SITES
    site_data = {s: load_site(args.artifacts, s) for s in all_sites}
    slots_per_day = int(86400 / cfg.solar.slot_seconds)
    trace_days = len(site_data[TRAIN_SITE]["harvest_j"]) * cfg.solar.slot_seconds / 86400.0
    print(f"[exp] loaded {len(all_sites)} site traces, {trace_days:.0f} days each")

    # -- fit every baseline ONCE on the training site -----------------------
    n_res, n_exit = bb["acc_grid"].shape
    idle_j = cfg.energy.p_idle_w * cfg.solar.slot_seconds
    wake_j = cfg.energy.e_wake_j

    lut = ReactiveLUT(cfg.control.soc_bins, n_res, n_exit)
    lut.fit(bb["energy_table"], bb["sense_energy"], bb["acc_grid"], idle_j, wake_j)
    print("[exp] fit ReactiveLUT (ePerceptive) on", TRAIN_SITE)

    harv_q = harv_acts = mdp = mdp_acts = mdp_edges = None
    if not args.skip_rl:
        t0 = time.time()
        harv_q, harv_acts = fit_harvsched(cfg, bb, site_data[TRAIN_SITE])
        print(f"[exp] fit HarvSched Q-learning on {TRAIN_SITE} ({time.time()-t0:.1f}s)")
        t0 = time.time()
        mdp, mdp_acts, mdp_edges = fit_monotone_mdp(cfg, bb, site_data[TRAIN_SITE])
        print(f"[exp] fit Bullo-style monotone MDP on {TRAIN_SITE} ({time.time()-t0:.1f}s)")

    # -- main comparison: every policy x every site -------------------------
    all_rows = []
    all_results = {}
    for site in all_sites:
        t0 = time.time()
        rows, results, t_ours = run_all_policies_for_site(
            cfg, bb, site, site_data[site], harv_q, harv_acts, mdp, mdp_acts, mdp_edges,
            lut, skip_rl=args.skip_rl)
        all_rows.extend(rows)
        all_results[site] = results
        dt = time.time() - t0
        summary = "  ".join(f"{r['policy']}={r['total_value']:.0f}" for r in rows)
        print(f"[exp] {site:8s} done in {dt:5.1f}s  {summary}")

    with open(f"{args.out}/tables/main_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"[exp] wrote {args.out}/tables/main_comparison.csv ({len(all_rows)} rows)")

    # ==== Experiment 1: value of forecast (train site) =====================
    print("[exp] Experiment 1: value of forecast ...")
    site_d = site_data[TRAIN_SITE]

    def run_variant(lower_j_per_day):
        return run_shadow_price(cfg, bb["energy_table"], bb["sense_energy"], bb["acc_grid"],
                                  bb["per_image"], site_d["harvest_j"], lower_j_per_day,
                                  slots_per_day, int(cfg.control.replan_minutes*60/cfg.solar.slot_seconds),
                                  site_d["stream"], cfg.battery_capacity_j,
                                  cfg.battery.soc_min_frac * cfg.battery_capacity_j,
                                  site_d["elevation"], confidence_aware=True)

    # perfect foresight of harvest (upper reference for the forecaster itself)
    perfect_daily = site_d["harvest_j"].reshape(-1, slots_per_day).sum(1)
    res_perfect = run_variant(perfect_daily)
    # our default: conformal lower bound (already computed in script 03)
    res_conformal = run_variant(site_d["lower_j_per_day"])
    # point forecast only, no conformal safety margin
    res_point = run_variant(site_d["point_j_per_day"])
    # naive: flat historical mean every day, no per-day adaptation at all
    naive_budget = np.full_like(site_d["lower_j_per_day"], site_d["point_j_per_day"].mean() * 0.8)
    res_naive = run_variant(naive_budget)

    v_reactive = float((site_d["stream"].value * all_results[TRAIN_SITE]["reactive"].correct).sum())
    v_oracle = float((site_d["stream"].value * all_results[TRAIN_SITE]["oracle"].correct).sum())

    vof_rows = []
    for name, res in [("perfect_foresight", res_perfect), ("conformal (ours)", res_conformal),
                       ("point_forecast_only", res_point), ("naive_flat_mean", res_naive)]:
        v = float((site_d["stream"].value * res.correct).sum())
        vof_rows.append(dict(variant=name, total_value=v,
                               value_of_forecast=value_of_forecast(v, v_reactive, v_oracle),
                               pct_of_oracle=percent_of_oracle(v, v_oracle),
                               downtime_frac=float(res.depleted.mean())))
    vof_rows.append(dict(variant="reactive_baseline", total_value=v_reactive,
                           value_of_forecast=0.0, pct_of_oracle=percent_of_oracle(v_reactive, v_oracle),
                           downtime_frac=float(all_results[TRAIN_SITE]["reactive"].depleted.mean())))
    vof_rows.append(dict(variant="oracle_ceiling", total_value=v_oracle, value_of_forecast=1.0,
                           pct_of_oracle=100.0,
                           downtime_frac=float(all_results[TRAIN_SITE]["oracle"].depleted.mean())))

    with open(f"{args.out}/tables/value_of_forecast.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vof_rows[0].keys()))
        w.writeheader(); w.writerows(vof_rows)
    print(f"[exp] wrote {args.out}/tables/value_of_forecast.csv")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = [r["variant"] for r in vof_rows]
    vals = [r["pct_of_oracle"] for r in vof_rows]
    colors = ["#C97A83", "#9A6508", "#8A5B12", "#6C7688", "#3B4453", "#151A23"]
    ax.barh(names, vals, color=colors[:len(names)])
    ax.set_xlabel("% of oracle (perfect-foresight) total value")
    ax.set_xlim(0, 105)
    ax.set_title(f"Value of forecast, {TRAIN_SITE} ({trace_days:.0f}-day trace)")
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/vof_sweep.png", dpi=150)
    plt.close(fig)
    print(f"[exp] wrote {args.out}/figures/vof_sweep.png")

    # ==== Experiment 2: risk-accuracy frontier (train site) =================
    print("[exp] Experiment 2: risk-accuracy frontier ...")
    alphas = [0.02, 0.05, 0.10, 0.20, 0.30]
    risk_rows = []
    for a in alphas:
        fc = simulate_forecast_and_calibration(
            site_d["kc"], slots_per_day, slots_per_day, site_d["ghi_clear"],
            cfg.solar.panel_area_m2, cfg.solar.panel_efficiency, cfg.solar.slot_seconds,
            a, cfg.control.aci_gamma,
        )
        res = run_variant(fc["lower_j"])
        v = float((site_d["stream"].value * res.correct).sum())
        risk_rows.append(dict(alpha=a, target_coverage=1 - a,
                                empirical_coverage=fc["aci"].empirical_coverage(),
                                total_value=v, pct_of_oracle=percent_of_oracle(v, v_oracle),
                                downtime_frac=float(res.depleted.mean())))
    with open(f"{args.out}/tables/risk_frontier.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(risk_rows[0].keys()))
        w.writeheader(); w.writerows(risk_rows)
    print(f"[exp] wrote {args.out}/tables/risk_frontier.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot([r["downtime_frac"]*100 for r in risk_rows], [r["pct_of_oracle"] for r in risk_rows],
             "o-", color="#9A6508")
    for r in risk_rows:
        ax1.annotate(f"a={r['alpha']}", (r["downtime_frac"]*100, r["pct_of_oracle"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax1.set_xlabel("downtime (% of slots below floor)")
    ax1.set_ylabel("% of oracle value")
    ax1.set_title("Risk-accuracy frontier")

    ax2.plot([1-r["alpha"] for r in risk_rows], [r["empirical_coverage"] for r in risk_rows],
             "o-", color="#116460", label="empirical")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="perfect calibration")
    ax2.set_xlabel("target coverage (1 - alpha)")
    ax2.set_ylabel("empirical coverage")
    ax2.set_xlim(0.65, 1.0); ax2.set_ylim(0.65, 1.0)
    ax2.legend(fontsize=8)
    ax2.set_title("Conformal calibration check")
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/risk_frontier.png", dpi=150)
    plt.close(fig)
    print(f"[exp] wrote {args.out}/figures/risk_frontier.png")

    # ==== Experiment 3: carbon ranking inversion (train site) ===============
    print("[exp] Experiment 3: carbon ranking inversion ...")
    battery = Battery(cfg.battery, cfg.battery_capacity_j).set_slot_seconds(cfg.solar.slot_seconds)
    wear_of_proxy = {k: v / cfg.battery_capacity_j for k, v in bb["energy_table"].items()}

    carbon_rows = []
    for mu_name, mu in [("aggressive (mu=0)", 0.0), ("wear-aware (mu tuned)", 8.0)]:
        ctrl_res = run_shadow_price(cfg, bb["energy_table"], bb["sense_energy"], bb["acc_grid"],
                                      bb["per_image"], site_d["harvest_j"], site_d["lower_j_per_day"],
                                      slots_per_day, int(cfg.control.replan_minutes*60/cfg.solar.slot_seconds),
                                      site_d["stream"], cfg.battery_capacity_j,
                                      cfg.battery.soc_min_frac * cfg.battery_capacity_j,
                                      site_d["elevation"], wear_of=wear_of_proxy if mu > 0 else None,
                                      confidence_aware=True)
        total_value = float((site_d["stream"].value * ctrl_res.correct).sum())
        total_energy = float(ctrl_res.energy_j.sum())
        wear_curve, half_cycles = battery.wear_from_soc_series(ctrl_res.soc_frac)
        n_repl = replacements_from_wear(wear_curve[-1] if len(wear_curve) else 0.0)
        carbon_kg = total_embodied_carbon_kg(cfg.carbon, cfg.battery.capacity_wh, n_repl)
        carbon_rows.append(dict(
            variant=mu_name, total_value=total_value, total_energy_j=total_energy,
            value_per_joule=total_value / max(total_energy, 1e-9),
            final_wear=wear_curve[-1] if len(wear_curve) else 0.0,
            n_replacements=n_repl, carbon_kg=carbon_kg,
            ctu=carbon_normalised_task_utility(total_value, carbon_kg),
        ))
    with open(f"{args.out}/tables/carbon_inversion.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(carbon_rows[0].keys()))
        w.writeheader(); w.writerows(carbon_rows)
    print(f"[exp] wrote {args.out}/tables/carbon_inversion.csv")
    for r in carbon_rows:
        print(f"    {r['variant']:24s} value/J={r['value_per_joule']:.4f}  CTU={r['ctu']:.3f}  "
              f"wear={r['final_wear']:.5f}  replacements={r['n_replacements']}")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9, 4))
    names = [r["variant"] for r in carbon_rows]
    axa.bar(names, [r["value_per_joule"] for r in carbon_rows], color=["#9B333E", "#116460"])
    axa.set_ylabel("value captured per joule"); axa.set_title("Ranked by energy efficiency")
    axa.tick_params(axis="x", rotation=15)
    axb.bar(names, [r["ctu"] for r in carbon_rows], color=["#9B333E", "#116460"])
    axb.set_ylabel("carbon-normalised task utility (CTU)"); axb.set_title("Ranked by lifetime carbon")
    axb.tick_params(axis="x", rotation=15)
    fig.suptitle("The energy-efficient policy is not the low-carbon policy")
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/carbon_inversion.png", dpi=150)
    plt.close(fig)
    print(f"[exp] wrote {args.out}/figures/carbon_inversion.png")

    # ==== Experiment 4: cross-climate transfer bar chart ====================
    print("[exp] Experiment 4: cross-climate transfer ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    policies_to_plot = ["ours", "reactive"] + ([] if args.skip_rl else ["qlearning", "mdp"]) + ["static_max"]
    width = 0.15
    x = np.arange(len(all_sites))
    for i, pol in enumerate(policies_to_plot):
        vals = []
        for site in all_sites:
            row = next(r for r in all_rows if r["policy"] == pol and r["site"] == site)
            vals.append(row["pct_of_oracle"])
        ax.bar(x + i * width, vals, width, label=pol, color=PALETTE.get(pol, None))
    ax.set_xticks(x + width * (len(policies_to_plot) - 1) / 2)
    ax.set_xticklabels([f"{s}\n{'(train)' if s == TRAIN_SITE else '(held-out)'}" for s in all_sites])
    ax.set_ylabel("% of oracle value")
    ax.set_title("Cross-climate transfer: fit on Dhaka, evaluate everywhere")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/cross_climate_transfer.png", dpi=150)
    plt.close(fig)
    print(f"[exp] wrote {args.out}/figures/cross_climate_transfer.png")

    # ==== Ablations ===========================================================
    print("[exp] Ablations: knob subsets + confidence-awareness ...")
    ablation_rows = []

    # confidence-aware vs -agnostic
    res_conf_agnostic = run_shadow_price(
        cfg, bb["energy_table"], bb["sense_energy"], bb["acc_grid"], bb["per_image"],
        site_d["harvest_j"], site_d["lower_j_per_day"], slots_per_day,
        int(cfg.control.replan_minutes*60/cfg.solar.slot_seconds), site_d["stream"],
        cfg.battery_capacity_j, cfg.battery.soc_min_frac * cfg.battery_capacity_j,
        site_d["elevation"], confidence_aware=False)
    v_conf_agnostic = float((site_d["stream"].value * res_conf_agnostic.correct).sum())
    v_ours_train = float((site_d["stream"].value * all_results[TRAIN_SITE]["ours"].correct).sum())
    ablation_rows.append(dict(ablation="confidence_aware_fast_loop", value=v_ours_train,
                                pct_of_oracle=percent_of_oracle(v_ours_train, v_oracle)))
    ablation_rows.append(dict(ablation="confidence_agnostic_fast_loop", value=v_conf_agnostic,
                                pct_of_oracle=percent_of_oracle(v_conf_agnostic, v_oracle)))

    # duty knob on/off: a "no duty" controller that must always wake (only
    # resolution + depth are adaptive) vs the full ours with sleep available.
    # Approximated by a fixed act_fn that never sleeps, downgrading to the
    # cheapest config under scarcity instead of skipping the slot entirely --
    # isolates what the duty knob itself is worth on top of the other two.
    n = len(site_d["harvest_j"])
    forced = run_fixed_policy(
        cfg, bb["energy_table"], bb["sense_energy"], bb["per_image"], site_d["harvest_j"],
        site_d["stream"], cfg.battery_capacity_j, cfg.battery.soc_min_frac * cfg.battery_capacity_j,
        lambda t, s: Action(1, 0, 0) if s < 0.15 else Action(1, n_res - 1, n_exit - 1),
    )
    v_forced = float((site_d["stream"].value * forced.correct).sum())
    ablation_rows.append(dict(ablation="knobs_res_and_depth_only_no_duty", value=v_forced,
                                pct_of_oracle=percent_of_oracle(v_forced, v_oracle)))
    ablation_rows.append(dict(ablation="knobs_res_depth_and_duty_ours", value=v_ours_train,
                                pct_of_oracle=percent_of_oracle(v_ours_train, v_oracle)))

    with open(f"{args.out}/tables/ablations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        w.writeheader(); w.writerows(ablation_rows)
    print(f"[exp] wrote {args.out}/tables/ablations.csv")

    # ==== Pareto frontier figure (train site, all policies) ==================
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for pol in ["static_min", "static_max", "conf_only", "reactive"] + \
               ([] if args.skip_rl else ["qlearning", "mdp"]) + ["ours", "oracle"]:
        row = next(r for r in all_rows if r["policy"] == pol and r["site"] == TRAIN_SITE)
        ax.scatter(row["total_energy_j"] / 1000, row["total_value"], s=90,
                   color=PALETTE.get(pol, "#888"), edgecolor="k", linewidth=0.6,
                   label=pol, zorder=3)
    ax.set_xlabel("total energy consumed (kJ)")
    ax.set_ylabel("total value captured")
    ax.set_title(f"Accuracy-energy Pareto picture, {TRAIN_SITE}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/pareto_frontier.png", dpi=150)
    plt.close(fig)
    print(f"[exp] wrote {args.out}/figures/pareto_frontier.png")

    # ==== SoC timeline example figure (train site, first 5 days) ============
    n_show = min(slots_per_day * 5, n)
    t_days = np.arange(n_show) / slots_per_day
    fig, ax = plt.subplots(figsize=(10, 4))
    for pol, res in [("ours", all_results[TRAIN_SITE]["ours"]),
                      ("reactive", all_results[TRAIN_SITE]["reactive"]),
                      ("static_max", all_results[TRAIN_SITE]["static_max"]),
                      ("oracle", all_results[TRAIN_SITE]["oracle"])]:
        ax.plot(t_days, res.soc_frac[:n_show] * 100, color=PALETTE.get(pol), label=pol, lw=1.3)
    ax.axhline(cfg.battery.soc_min_frac * 100, color="#9B333E", ls="--", lw=1, label="floor")
    ax.set_xlabel("day"); ax.set_ylabel("state of charge (%)")
    ax.set_title(f"SoC trajectory, {TRAIN_SITE}, first 5 days")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/soc_timeline_example.png", dpi=150)
    plt.close(fig)
    print(f"[exp] wrote {args.out}/figures/soc_timeline_example.png")

    print("[exp] done.")


if __name__ == "__main__":
    main()
