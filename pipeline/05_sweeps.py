#!/usr/bin/env python
"""Frontiers, regime curves and sensitivity to every assumed constant.

    python pipeline/05_sweeps.py                    # all three
    python pipeline/05_sweeps.py --only frontier regime

frontier     each tunable policy traced across its own knob, so policies are
             compared as accuracy-vs-battery-life curves, not single points
regime       key policies across the harvest-to-demand ratio
sensitivity  one assumption at a time pushed to both ends of its range. A
             conclusion that flips inside these ranges is not a result.
autonomy     how many days of load the battery holds, which decides whether
             scheduling within a day can matter at all
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sunsched.cli import banner, base_parser, load_cfg, write_csv
from sunsched.eval.stats import aggregate
from sunsched.experiment import make_jobs, run_jobs

METRICS = ["value_score", "animal_recall", "dropped_frac", "latency_p95_h",
           "battery_years_to_eol", "frac_time_soc_above_90", "b_wakes_per_day", "calendar_share"]
KEY_POLICIES = ["always_now", "ee_now", "ceiling_now", "lazy_defer", "sunsched"]

# (config field, low, high). Ranges are broad on purpose: none of these
# numbers is measured in this project.
SENSITIVITY = [
    ("node.tier_b_wake_j", 3.0, 30.0),
    ("node.tier_b_active_w", 1.5, 6.0),
    ("node.e_capture_night_j", 0.4, 3.0),
    ("node.p_sleep_w", 0.0005, 0.01),
    ("node.tier_a_pj_per_mac", 5.0, 100.0),
    ("battery.capacity_wh", 5.0, 40.0),
    ("solar.enclosure_gain_c", 0.0, 30.0),
    ("aging.k_sigma", 0.52, 1.56),
    ("aging.k_T", 0.035, 0.104),
    ("aging.k_t", 2.07e-10, 6.21e-10),
    ("control.deadline_h", 12.0, 96.0),
    # When the nightly reserve is small against the battery, the charge ceiling
    # sits on this floor and the conformal bound never binds (sunsched and
    # sunsched_no_conformal then coincide). Sweeping it shows whether the
    # risk-controlled part of the method does any work.
    ("control.min_ceiling", 0.15, 0.5),
]

# Days of load the battery holds. At this node's duty cycle the default 10 Wh
# is about 175 days, where one night's reserve is 0.3% of the store: the charge
# ceiling clips to control.min_ceiling, the cell turns under one equivalent full
# cycle a year and ages by calendar, and nothing that schedules within a day can
# differ from simply charging to that floor. Note that the capacity_wh row of
# SENSITIVITY (5-40 Wh) never leaves that regime either: even 5 Wh is ~87 days.
# The whole curve is reported, not a chosen point on it.
AUTONOMY_DAYS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 175.0)
AUTONOMY_POLICIES = KEY_POLICIES + ["sunsched_no_conformal", "sunsched_no_ceiling"]
# Carried alongside METRICS so the curve shows not just whether sunsched wins,
# but whether mechanism 3 was ever able to bind.
AUTONOMY_METRICS = ["ceiling_at_floor_frac", "reserve_coverage", "soc_mean",
                    "equiv_full_cycles_per_year", "capacity_wh", "days_of_autonomy"]


def frontier(cfg):
    ratios = cfg.experiment.ratios
    jobs = []
    for a in cfg.experiment.alpha_sweep:
        jobs += make_jobs(cfg, ["sunsched"], ratios, overrides={"control.alpha": a},
                          tags=dict(sweep="alpha", value=a))
    for m in (0.0, 0.03, 0.1, 0.2):
        jobs += make_jobs(cfg, ["sunsched"], ratios, overrides={"control.reserve_margin": m},
                          tags=dict(sweep="reserve_margin", value=m))
    for th in (0.5, 0.7, 0.9, 0.99):
        jobs += make_jobs(cfg, ["ee_now"], ratios, overrides={"baselines.ee_theta_hi": th},
                          tags=dict(sweep="ee_theta_hi", value=th))
    for m in (0.05, 0.1, 0.2, 0.4):
        jobs += make_jobs(cfg, ["ceiling_now"], ratios, overrides={"baselines.ceiling_margin": m},
                          tags=dict(sweep="ceiling_margin", value=m))
    for th in (0.5, 0.7, 0.9, 0.99):
        jobs += make_jobs(cfg, ["lazy_defer"], ratios, overrides={"baselines.lazy_theta": th},
                          tags=dict(sweep="lazy_theta", value=th))
    rows = run_jobs(jobs, cfg.experiment.n_workers, "frontier")
    write_csv(f"{cfg.results_dir}/tables/frontier_runs.csv", rows)
    ok = [r for r in rows if "error" not in r]
    agg = aggregate(ok, ["ratio", "policy", "tag_sweep", "tag_value"], METRICS,
                    cfg.experiment.ci_level, cfg.experiment.n_bootstrap)
    write_csv(f"{cfg.results_dir}/tables/frontier.csv", agg)


def regime(cfg):
    jobs = make_jobs(cfg, KEY_POLICIES, cfg.experiment.ratio_sweep, tags=dict(sweep="ratio"))
    rows = run_jobs(jobs, cfg.experiment.n_workers, "regime")
    write_csv(f"{cfg.results_dir}/tables/regime_runs.csv", rows)
    ok = [r for r in rows if "error" not in r]
    write_csv(f"{cfg.results_dir}/tables/regime.csv",
              aggregate(ok, ["ratio", "policy"], METRICS, cfg.experiment.ci_level,
                        cfg.experiment.n_bootstrap))


def sensitivity(cfg):
    ratio = [cfg.experiment.ratios[0]]
    jobs = []
    for field, lo, hi in SENSITIVITY:
        for level, value in (("low", lo), ("high", hi)):
            jobs += make_jobs(cfg, KEY_POLICIES, ratio, overrides={field: value},
                              tags=dict(param=field, level=level, value=value))
    from sunsched.config import SITES
    for site in SITES:
        if site != cfg.experiment.main_site:
            jobs += make_jobs(cfg, KEY_POLICIES, ratio, site=site,
                              tags=dict(param="site", level=site, value=0))
    rows = run_jobs(jobs, cfg.experiment.n_workers, "sensitivity")
    write_csv(f"{cfg.results_dir}/tables/sensitivity_runs.csv", rows)
    ok = [r for r in rows if "error" not in r]
    write_csv(f"{cfg.results_dir}/tables/sensitivity.csv",
              aggregate(ok, ["tag_param", "tag_level", "policy"], METRICS,
                        cfg.experiment.ci_level, cfg.experiment.n_bootstrap))


def autonomy(cfg):
    """Does scheduling in time ever pay, and below what battery size?

    The pre-registered kill test ran at one sizing, which turned out to sit far
    inside the regime where no schedule can help. This sweep is exploratory: it
    reports every autonomy level rather than selecting one, so it answers where
    the method could work without tuning anything to make it pass.
    """
    jobs = []
    for d in AUTONOMY_DAYS:
        jobs += make_jobs(cfg, AUTONOMY_POLICIES, cfg.experiment.ratios, autonomy_days=d,
                          tags=dict(sweep="autonomy_days", value=d))
    rows = run_jobs(jobs, cfg.experiment.n_workers, "autonomy")
    write_csv(f"{cfg.results_dir}/tables/autonomy_runs.csv", rows)
    ok = [r for r in rows if "error" not in r]
    write_csv(f"{cfg.results_dir}/tables/autonomy.csv",
              aggregate(ok, ["tag_value", "ratio", "policy"], METRICS + AUTONOMY_METRICS,
                        cfg.experiment.ci_level, cfg.experiment.n_bootstrap))


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--only", nargs="*",
                    default=["frontier", "regime", "sensitivity", "autonomy"])
    args = ap.parse_args()
    cfg = load_cfg(args)
    for name in args.only:
        banner(f"sweep: {name}")
        {"frontier": frontier, "regime": regime, "sensitivity": sensitivity,
         "autonomy": autonomy}[name](cfg)


if __name__ == "__main__":
    main()
