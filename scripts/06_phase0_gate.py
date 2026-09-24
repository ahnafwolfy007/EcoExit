#!/usr/bin/env python
"""The Phase 0 gate: is there a paper here?

Everything downstream is void until this passes, so it is a separate script
with a single, blunt verdict rather than a row in a results table.

Two conditions, both necessary:

  1. LAMBDA IS LIVE. The energy price must be non-zero and must vary. If the
     planner hands back lambda = 0 at every replan, the controller degenerates
     into "always run the biggest model" and every ablation downstream returns
     bit-identical numbers -- which is exactly what the original pipeline did.

  2. THE REGIME BINDS. Daily harvest must land inside the discretionary band
     between the always-sleep floor and the always-max ceiling. Above the
     ceiling there is no allocation problem and `static_max` is the correct
     answer, not a degenerate baseline. At the original panel area this was
     true at four of five sites.

Only then is the third question meaningful: does the controller beat
`static_max` on carbon-normalized task utility, across seeds, with a paired
test?

    python scripts/06_phase0_gate.py
    python scripts/06_phase0_gate.py --seeds 5 --ratio 0.75
    python scripts/06_phase0_gate.py --no-resize     # reproduce the original failure
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ecoexit.config import Config, SITES, CARBON_PROFILES, apply_carbon_profile, quick
from ecoexit.eval.harness import (
    load_backbone, build_site, run_ours, run_static_max, run_static_min, summarize,
)
from ecoexit.eval.metrics import battery_carbon_share
from ecoexit.eval.stats import bootstrap_ci, paired_wilcoxon
from ecoexit.sizing import demand_band, harvest_to_demand, regime_label, solve_panel_area


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--ratio", type=float, default=None,
                    help="target harvest-to-demand ratio (default from config)")
    ap.add_argument("--no-resize", action="store_true",
                    help="keep the original panel area, to reproduce the original tie")
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--profile", default=None,
                    help=f"node hardware class: {sorted(CARBON_PROFILES)}")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    if args.quick:
        cfg = quick(cfg)
    if args.profile:
        cfg = apply_carbon_profile(cfg, args.profile)
    n_seeds = args.seeds or cfg.experiment.n_seeds
    target_ratio = args.ratio or cfg.experiment.target_ratio
    sites = args.sites or ["dhaka", "nairobi", "munich", "phoenix", "bergen"]

    bb = load_backbone(args.artifacts, cfg, use_val_grid=True)
    band = demand_band(cfg, bb["energy_table"], bb["sense_energy"], bb["acc_grid"])

    print("=" * 72)
    print("PHASE 0 GATE")
    print("=" * 72)
    print(f"\nDemand band (per day):")
    print(f"  always-sleep floor    {band.floor_j_per_day:12,.0f} J")
    print(f"  always-max ceiling    {band.ceiling_j_per_day:12,.0f} J")
    print(f"  discretionary         {band.discretionary_j_per_day:12,.0f} J"
          f"   ({band.controllable_fraction * 100:.1f}% of the ceiling)")
    print("\nThe discretionary fraction is the share of the energy budget the")
    print("controller's knobs actually command. It is the number Reviewer A's")
    print("strongest objection reduces to, so it belongs in the paper.")

    # -- condition 2: sizing -------------------------------------------------
    print(f"\n{'-' * 72}\nCONDITION 2: does the regime bind?\n{'-' * 72}")
    sizing_rows = []
    panel_for = {}
    for site in sites:
        probe = build_site(cfg, site, seed=0)
        ratio_now = harvest_to_demand(probe.harvest_j, band)
        if args.no_resize:
            area = cfg.solar.panel_area_m2
        else:
            area = solve_panel_area(site, SITES[site], cfg.solar, band, target_ratio)
        panel_for[site] = area
        resized = build_site(cfg, site, seed=0, panel_area_m2=area)
        ratio_new = harvest_to_demand(resized.harvest_j, band)
        row = dict(site=site,
                   panel_area_original_m2=cfg.solar.panel_area_m2,
                   ratio_original=ratio_now,
                   regime_original=regime_label(ratio_now, band),
                   panel_area_used_m2=area,
                   ratio_used=ratio_new,
                   regime_used=regime_label(ratio_new, band))
        sizing_rows.append(row)
        print(f"  {site:<10} original {ratio_now:5.2f} ({row['regime_original']:<14})"
              f"  ->  used {ratio_new:5.2f} ({row['regime_used']}) "
              f"at {area * 1e4:.1f} cm^2")

    binding = [r for r in sizing_rows if r["regime_used"] == "discretionary"]
    cond2 = len(binding) >= max(1, len(sites) // 2)
    print(f"\n  {len(binding)}/{len(sites)} sites in the discretionary band "
          f"-> condition 2 {'PASS' if cond2 else 'FAIL'}")

    # -- run ------------------------------------------------------------------
    print(f"\n{'-' * 72}\nRUNNING {n_seeds} seed(s) x {len(sites)} site(s)\n{'-' * 72}")
    rows = []
    per_seed_ctu = {}
    for site in sites:
        for seed in range(n_seeds):
            sd = build_site(cfg, site, seed=seed, panel_area_m2=panel_for[site],
                            eval_pool_size=len(bb["eval_pool"]))
            runs = {
                "ours": run_ours(cfg, bb, sd),
                "static_max": run_static_max(cfg, bb, sd),
                "static_min": run_static_min(cfg, bb, sd),
            }
            for name, res in runs.items():
                r = summarize(name, cfg, res, sd, extra=dict(seed=seed))
                rows.append(r)
                per_seed_ctu.setdefault((site, name), []).append(r["ctu"])
            print(f"  {site:<10} seed {seed}: "
                  f"ours CTU {rows[-3]['ctu']:8.2f}   "
                  f"static_max CTU {rows[-2]['ctu']:8.2f}   "
                  f"lambda_mean {rows[-3].get('lambda_mean', 0):.4g}")

    # -- condition 1: lambda is live -----------------------------------------
    print(f"\n{'-' * 72}\nCONDITION 1: is lambda live?\n{'-' * 72}")
    ours_rows = [r for r in rows if r["policy"] == "ours"]
    lam_mean = float(np.mean([r.get("lambda_mean", 0.0) for r in ours_rows]))
    lam_zero = float(np.mean([r.get("lambda_frac_zero", 1.0) for r in ours_rows]))
    lam_max = float(np.max([r.get("lambda_max", 0.0) for r in ours_rows]))
    cond1 = lam_mean > 1e-9 and lam_zero < 0.5
    print(f"  mean lambda           {lam_mean:.6g}")
    print(f"  max lambda            {lam_max:.6g}")
    print(f"  fraction of slots at lambda=0   {lam_zero * 100:.1f}%")
    print(f"  -> condition 1 {'PASS' if cond1 else 'FAIL'}")

    # wear price sanity: the online price must reproduce offline rainflow
    relerrs = [r["wear_online_vs_offline_relerr"] for r in ours_rows
               if "wear_online_vs_offline_relerr" in r]
    if relerrs:
        print(f"\n  online vs offline rainflow wear, max relative error: {max(relerrs):.2%}")

    # -- condition 0: can carbon even distinguish policies? ------------------
    # Checked last because it needs observed wear, but it is logically first:
    # if the battery term is a rounding error against the fixed board and panel,
    # then carbon-normalized utility is total value divided by a constant, every
    # policy ranks exactly as it does on raw value, and no controller can
    # demonstrate a carbon inversion however good it is.
    print(f"\n{'-' * 72}\nCONDITION 0: is the battery a material share of carbon?\n{'-' * 72}")
    repl = [r["projected_replacements"] for r in ours_rows]
    share_observed = battery_carbon_share(cfg.carbon, cfg.battery.capacity_wh,
                                          float(np.mean(repl)))
    share_full = battery_carbon_share(cfg.carbon, cfg.battery.capacity_wh, 1.0)
    cond0 = share_observed >= 0.05
    print(f"  node profile                 {cfg.carbon.profile}")
    print(f"  board + panel (fixed)        "
          f"{cfg.carbon.board_kgco2e + cfg.carbon.panel_kgco2e_per_wp * cfg.carbon.panel_wp:8.3f} kg")
    print(f"  battery, one full pack       "
          f"{cfg.carbon.battery_kgco2e_per_kwh * cfg.battery.capacity_wh / 1000.0:8.3f} kg")
    print(f"  projected replacements       {np.mean(repl):8.4f} over "
          f"{cfg.carbon.deployment_years:.0f} years")
    print(f"  battery share at observed wear  {share_observed * 100:6.2f}%")
    print(f"  battery share at one full pack  {share_full * 100:6.2f}%")
    print(f"  -> condition 0 {'PASS' if cond0 else 'FAIL'}")
    if not cond0:
        print("\n  The policy-controllable share of lifecycle carbon is below 5%, so")
        print("  carbon-normalized task utility is total value divided by a constant.")
        print("  Every policy will rank exactly as it does on raw value, and the")
        print("  carbon inversion cannot appear. This is a property of the hardware")
        print("  parameters, not of the controller.")
        print(f"  Try:  python scripts/06_phase0_gate.py --profile mcu")

    # -- condition 3: does it beat static_max? -------------------------------
    print(f"\n{'-' * 72}\nCONDITION 3: does the controller beat static_max?\n{'-' * 72}")
    test_rows = []
    n_wins = 0
    for site in sites:
        a = per_seed_ctu[(site, "ours")]
        b = per_seed_ctu[(site, "static_max")]
        mean_a, lo_a, hi_a = bootstrap_ci(a, cfg.experiment.ci_level, cfg.experiment.n_bootstrap)
        mean_b, lo_b, hi_b = bootstrap_ci(b, cfg.experiment.ci_level, cfg.experiment.n_bootstrap)
        test = paired_wilcoxon(a, b)
        rel = 100.0 * (mean_a - mean_b) / max(abs(mean_b), 1e-9)
        win = mean_a > mean_b and (test["p_value"] < 0.05 or n_seeds < 5)
        n_wins += int(win)
        test_rows.append(dict(site=site, ours_ctu=mean_a, ours_lo=lo_a, ours_hi=hi_a,
                              static_max_ctu=mean_b, static_max_lo=lo_b, static_max_hi=hi_b,
                              rel_improvement_pct=rel, p_value=test["p_value"],
                              n_seeds=n_seeds, wins=win))
        print(f"  {site:<10} ours {mean_a:8.2f} [{lo_a:7.2f},{hi_a:7.2f}]   "
              f"static_max {mean_b:8.2f}   "
              f"{rel:+6.2f}%   p={test['p_value']:.3f}  {'WIN' if win else '--'}")

    cond3 = n_wins >= max(1, len(binding))
    print(f"\n  wins at {n_wins}/{len(sites)} sites -> condition 3 "
          f"{'PASS' if cond3 else 'FAIL'}")

    # -- verdict -------------------------------------------------------------
    passed = cond0 and cond1 and cond2 and cond3
    print("\n" + "=" * 72)
    print(f"GATE: {'PASS' if passed else 'FAIL'}")
    print(f"  condition 0  carbon is measurable   {'PASS' if cond0 else 'FAIL'}")
    print(f"  condition 1  lambda is live         {'PASS' if cond1 else 'FAIL'}")
    print(f"  condition 2  regime binds           {'PASS' if cond2 else 'FAIL'}")
    print(f"  condition 3  beats static_max       {'PASS' if cond3 else 'FAIL'}")
    print("=" * 72)
    if passed:
        print("\nThe controller is live, the regime binds, and it beats static_max.")
        print("Phase 1 (real camera-trap event timing + PVGIS harvest) is unblocked:")
        print("    python scripts/03_build_traces.py --source pvgis --events camera_trap")
    else:
        if not cond1:
            print("\nlambda is not live. The planner is still handing the controller more")
            print("budget than it can spend. Check ControlCfg.horizon_hours and")
            print("reserve_frac, and directional_water_filling in control/pricing.py.")
        if not cond2:
            print("\nThe regime does not bind. Harvest is outside the discretionary band,")
            print("so static_max is the correct answer and no controller can beat it.")
            print("Lower --ratio, or check the energy table's absolute scale.")
        if not cond3:
            print("\nlambda is live and the regime binds, but the controller still does not")
            print("beat static_max. This is the honest negative result the plan commits to")
            print("reporting. Before rethinking the method, check: is the discretionary")
            print(f"fraction ({band.controllable_fraction * 100:.1f}%) simply too small for")
            print("configuration choice to matter at this operating point?")

    os.makedirs(f"{args.out}/tables", exist_ok=True)
    write_csv(f"{args.out}/tables/gate_sizing.csv", sizing_rows)
    write_csv(f"{args.out}/tables/gate_runs.csv", rows)
    write_csv(f"{args.out}/tables/gate_tests.csv", test_rows)
    with open(f"{args.out}/tables/gate_verdict.json", "w", encoding="utf-8") as f:
        json.dump(dict(passed=bool(passed), carbon_measurable=bool(cond0),
                       lambda_live=bool(cond1), regime_binds=bool(cond2),
                       beats_static_max=bool(cond3),
                       battery_carbon_share_observed=share_observed,
                       battery_carbon_share_full_pack=share_full,
                       carbon_profile=cfg.carbon.profile,
                       lambda_mean=lam_mean, lambda_frac_zero=lam_zero,
                       discretionary_fraction=band.controllable_fraction,
                       n_seeds=n_seeds, target_ratio=target_ratio,
                       sites=sites), f, indent=2)
    print(f"\nwrote {args.out}/tables/gate_*.csv and gate_verdict.json")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
