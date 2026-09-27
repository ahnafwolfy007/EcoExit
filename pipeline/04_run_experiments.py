#!/usr/bin/env python
"""Main comparison: every policy x every held-out camera x every weather year x each regime.

    python pipeline/04_run_experiments.py

Writes outputs/results/tables/runs.csv (one row per simulated deployment),
summary.csv (mean and 95% bootstrap CI per policy and regime), and tests.csv
(paired Wilcoxon tests of SunSched against every other policy, Holm-corrected).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sunsched.cli import banner, base_parser, load_cfg, write_csv
from sunsched.eval.stats import aggregate, holm, paired_wilcoxon
from sunsched.experiment import ABLATIONS, BASELINES, OURS, make_jobs, run_jobs

KEY_METRICS = ["value_score", "gain_captured", "accuracy", "macro_f1", "animal_recall",
               "dropped_frac", "refined_now_frac", "refined_later_frac", "latency_mean_h",
               "latency_p95_h", "b_wakes_per_day", "soc_mean", "frac_time_soc_above_90",
               "battery_temp_mean_c", "f_calendar_per_year", "f_cycle_per_year",
               "calendar_share", "battery_years_to_eol", "replacements_per_deployment",
               "battery_kgco2e_per_deployment", "curtailed_kj", "dead_frac", "reserve_coverage"]
TEST_METRICS = ["value_score", "battery_years_to_eol", "animal_recall", "dropped_frac"]


def paired_tests(rows, reference="sunsched"):
    out = []
    for ratio in sorted({r["ratio"] for r in rows}):
        rr = [r for r in rows if r["ratio"] == ratio and "error" not in r]
        others = sorted({r["policy"] for r in rr} - {reference})
        for metric in TEST_METRICS:
            ref = {(r["location"], r["year"]): r[metric] for r in rr if r["policy"] == reference}
            tests = []
            for p in others:
                oth = {(r["location"], r["year"]): r[metric] for r in rr if r["policy"] == p}
                keys = sorted(set(ref) & set(oth))
                t = paired_wilcoxon([ref[k] for k in keys], [oth[k] for k in keys])
                tests.append(dict(ratio=ratio, metric=metric, policy=reference, versus=p, **t))
            for t, h in zip(tests, holm([t["p_value"] for t in tests])):
                t.update(h)
            out.extend(tests)
    return out


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--policies", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args)

    policies = args.policies or (BASELINES + OURS + ABLATIONS)
    banner(f"main comparison at site {cfg.experiment.main_site}, regimes {list(cfg.experiment.ratios)}")
    jobs = make_jobs(cfg, policies, cfg.experiment.ratios)
    rows = run_jobs(jobs, cfg.experiment.n_workers, "main")
    tables = f"{cfg.results_dir}/tables"
    write_csv(f"{tables}/runs.csv", rows)

    ok = [r for r in rows if "error" not in r]
    summary = aggregate(ok, ["ratio", "policy"], KEY_METRICS, cfg.experiment.ci_level,
                        cfg.experiment.n_bootstrap)
    write_csv(f"{tables}/summary.csv", summary)
    write_csv(f"{tables}/tests.csv", paired_tests(ok))

    for ratio in cfg.experiment.ratios:
        print(f"\n  harvest-to-demand ratio {ratio}")
        print(f"  {'policy':<24}{'value':>8}{'recall':>8}{'dropped':>8}{'lat p95 h':>10}"
              f"{'batt yrs':>10}{'SoC>90%':>9}{'wakes/d':>8}")
        for s in sorted((s for s in summary if s["ratio"] == ratio), key=lambda s: -s["value_score"]):
            print(f"  {s['policy']:<24}{s['value_score']:8.3f}{s['animal_recall']:8.3f}"
                  f"{s['dropped_frac']:8.3f}{s['latency_p95_h']:10.1f}{s['battery_years_to_eol']:10.2f}"
                  f"{s['frac_time_soc_above_90'] * 100:8.1f}%{s['b_wakes_per_day']:8.2f}")
    print(f"\n  wrote {tables}/runs.csv, summary.csv, tests.csv")


if __name__ == "__main__":
    main()
