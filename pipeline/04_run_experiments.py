#!/usr/bin/env python
"""Main experiment: every policy across the battery-autonomy x harvest-ratio grid.

    python pipeline/04_run_experiments.py --corpus cct20

Baselines and SunSched v2 run in every cell; the ablations run at the probe
cells only. Replicates are every eval camera x every weather year, paired
across policies.

Writes results/tables/runs.csv (one row per simulated deployment),
summary.csv (mean and 95% bootstrap CI per cell and policy), and
run_manifest.json (git commit and full configuration: the record that the
test-corpus run used the frozen, pre-registered setup).
"""
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sunsched.cli import banner, base_parser, load_cfg, write_csv, write_json
from sunsched.eval.stats import aggregate
from sunsched.experiment import (ABLATIONS, BASELINES, OURS, config_to_dict, grid_cells,
                                 make_jobs, run_jobs)

KEY_METRICS = ["value_score", "gain_captured", "accuracy", "macro_f1", "animal_recall",
               "dropped_frac", "refined_now_frac", "refined_later_frac", "latency_mean_h",
               "latency_p95_h", "b_wakes_per_day", "soc_mean", "frac_time_soc_above_90",
               "f_calendar_per_year", "f_cycle_per_year", "calendar_share",
               "battery_years_to_eol", "replacements_per_deployment", "curtailed_kj",
               "dead_frac", "reserve_coverage", "ceiling_mean"]


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return dict(commit=run("rev-parse", "HEAD") or "unknown",
                dirty=bool(run("status", "--porcelain", "--", "sunsched", "pipeline")))


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--policies", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args)
    tables = f"{cfg.results_dir}/tables"

    state = git_state()
    write_json(f"{tables}/run_manifest.json", dict(
        started=datetime.now().isoformat(timespec="seconds"), corpus=cfg.data.corpus,
        quick=bool(args.quick), git=state, config=config_to_dict(cfg)))
    if cfg.data.corpus == "serengeti" and state["dirty"]:
        print("WARNING: sunsched/ or pipeline/ has uncommitted changes. PREREGISTRATION.md requires\n"
              "the test-corpus run to use committed code; commit first or report this run as exploratory.")

    main_policies = args.policies or (BASELINES + OURS)
    banner(f"{cfg.data.corpus}: {len(main_policies)} policies x {len(grid_cells(cfg))} cells")
    rows = run_jobs(make_jobs(cfg, main_policies, grid_cells(cfg)), cfg.experiment.n_workers, "grid")
    if not args.policies:
        banner(f"ablations at probe cells {list(cfg.experiment.probe_cells)}")
        rows += run_jobs(make_jobs(cfg, ABLATIONS, cfg.experiment.probe_cells),
                         cfg.experiment.n_workers, "ablations")
    write_csv(f"{tables}/runs.csv", rows)

    ok = [r for r in rows if "error" not in r]
    summary = aggregate(ok, ["autonomy", "ratio", "policy"], KEY_METRICS, cfg.experiment.ci_level,
                        cfg.experiment.n_bootstrap)
    write_csv(f"{tables}/summary.csv", summary)

    print(f"\n  value score / battery years, mean over cameras x years")
    pols = [p for p in BASELINES + OURS if any(s["policy"] == p for s in summary)]
    print(f"  {'autonomy':>8} {'ratio':>5} " + "".join(f"{p[:13]:>15}" for p in pols))
    for a in cfg.experiment.autonomy_days:
        for r in cfg.experiment.ratios:
            cells = {s["policy"]: s for s in summary if s["autonomy"] == a and s["ratio"] == r}
            print(f"  {a:>8} {r:>5} " + "".join(
                f"{cells[p]['value_score']:>8.3f}/{cells[p]['battery_years_to_eol']:<5.1f}" if p in cells else f"{'-':>15}"
                for p in pols))
    print(f"\n  wrote {tables}/runs.csv, summary.csv, run_manifest.json")


if __name__ == "__main__":
    main()
