#!/usr/bin/env python
"""Sensitivity of the regime claims to every assumed constant.

    python pipeline/05_sweeps.py --corpus cct20
    python pipeline/05_sweeps.py --corpus cct20 --only sensitivity

sensitivity  each assumption pushed to both ends of its range, at the probe
             cells (one small-storage and one large-storage cell at least). A
             regime claim that flips inside these ranges is not a result.
sites        the same real animal activity under other climates.
alpha        SunSched v2's conformal risk level.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sunsched.cli import banner, base_parser, load_cfg, write_csv
from sunsched.config import SITES
from sunsched.eval.stats import aggregate
from sunsched.experiment import make_jobs, run_jobs

METRICS = ["value_score", "animal_recall", "dropped_frac", "latency_p95_h",
           "battery_years_to_eol", "frac_time_soc_above_90", "b_wakes_per_day", "calendar_share"]
KEY_POLICIES = ["always_now", "ceiling_now", "lazy_defer", "lazy_animal", "sunsched"]

# (config field, low, high). Ranges are broad on purpose: none of these is
# measured in this project.
SENSITIVITY = [
    ("node.tier_b_wake_j", 3.0, 30.0),
    ("node.tier_b_active_w", 1.5, 6.0),
    ("node.e_capture_night_j", 0.4, 3.0),
    ("node.p_sleep_w", 0.0005, 0.01),
    ("node.tier_a_pj_per_mac", 5.0, 100.0),
    ("solar.enclosure_gain_c", 0.0, 30.0),
    ("aging.k_sigma", 0.52, 1.56),
    ("aging.k_T", 0.035, 0.104),
    ("aging.k_t", 2.07e-10, 6.21e-10),
    ("control.deadline_h", 12.0, 96.0),
    ("control.min_ceiling", 0.15, 0.5),
    ("battery.soc_floor", 0.02, 0.15),
]


def run_group(cfg, name, jobs, keys):
    rows = run_jobs(jobs, cfg.experiment.n_workers, name)
    write_csv(f"{cfg.results_dir}/tables/{name}_runs.csv", rows)
    ok = [r for r in rows if "error" not in r]
    write_csv(f"{cfg.results_dir}/tables/{name}.csv",
              aggregate(ok, keys, METRICS, cfg.experiment.ci_level, cfg.experiment.n_bootstrap))


def sensitivity(cfg):
    jobs = []
    for field, lo, hi in SENSITIVITY:
        for level, value in (("low", lo), ("high", hi)):
            jobs += make_jobs(cfg, KEY_POLICIES, cfg.experiment.probe_cells, overrides={field: value},
                              tags=dict(param=field, level=level, value=value))
    run_group(cfg, "sensitivity", jobs, ["tag_param", "tag_level", "autonomy", "ratio", "policy"])


def sites(cfg):
    jobs = []
    for site in SITES:
        if site != cfg.experiment.main_site:
            jobs += make_jobs(cfg, KEY_POLICIES, cfg.experiment.probe_cells, site=site,
                              tags=dict(param="site", level=site, value=0))
    run_group(cfg, "sites", jobs, ["tag_level", "autonomy", "ratio", "policy"])


def alpha(cfg):
    jobs = []
    for a in cfg.experiment.alpha_sweep:
        jobs += make_jobs(cfg, ["sunsched"], cfg.experiment.probe_cells, overrides={"control.alpha": a},
                          tags=dict(param="control.alpha", level=str(a), value=a))
    run_group(cfg, "alpha", jobs, ["tag_value", "autonomy", "ratio", "policy"])


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--only", nargs="*", default=["sensitivity", "sites", "alpha"])
    args = ap.parse_args()
    cfg = load_cfg(args)
    for name in args.only:
        banner(f"sweep: {name}")
        {"sensitivity": sensitivity, "sites": sites, "alpha": alpha}[name](cfg)


if __name__ == "__main__":
    main()
