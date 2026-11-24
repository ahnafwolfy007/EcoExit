#!/usr/bin/env python
"""Experiments added in response to the paper review (run after stages 1-3).

    python pipeline/08_revision.py --out ./outputs              # every group
    python pipeline/08_revision.py --out ./outputs --only grid sens
    python pipeline/08_revision.py --out ./outputs --dry-run    # count jobs only

Uses the artifacts of the earlier full run (outputs/artifacts), so nothing is
downloaded and stages 1-3 are not repeated. Every table goes to
<out>/results/tables/revision/ and no earlier table is overwritten.

Groups (one line each; the reason is in each function's docstring):
  coverage  reserve rules compared on the same nights: ACI, split conformal,
            empirical quantile, point forecast, at alpha in ALPHAS
  grid      the full autonomy x ratio grid with the paper's policies plus the
            new pre-specified controls (defer_ceiling, batch_anytime,
            ceiling_conformal, defer_ceiling_conformal)
  sens      every assumed hardware constant at both ends of its range, at the
            four cells that carry the paper's claims
  wake      dose-response in the tier-B wake energy (where deferral stops paying)
  latency   deferral deadline and on-node queue size vs value and delay
  sites     the four claim cells under five other climates
  prereg    the grid of PREREGISTRATION.md (autonomy 0.5-100 d x ratio 0.75-3),
            used for the held-out Snapshot Serengeti test (--corpus serengeti)

Every run also scores its battery trace under k_sigma in KSIG and k_T in KT
(no extra simulation: see sunsched/env/battery.py:aging_variants).
"""
import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sunsched.cli import banner, base_parser, load_cfg, write_csv
from sunsched.config import SITES
from sunsched.experiment import config_to_dict, make_jobs, run_jobs

# The paper's grid (stage 5 'autonomy' sweep).
AUTONOMY = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 175.0)
RATIOS = (0.5, 1.0, 2.0)
# The cells the paper's two claims rest on: deferral at small storage,
# charge ceiling at large storage.
CLAIM_CELLS = ((0.5, 2.0), (1.0, 1.0), (30.0, 2.0), (100.0, 2.0))

PAPER_POLICIES = ["always_now", "ee_now", "ceiling_now", "lazy_defer"]
NEW_POLICIES = ["defer_ceiling", "batch_anytime", "ceiling_conformal", "defer_ceiling_conformal"]
CORE = ["always_now", "lazy_defer", "ceiling_now", "defer_ceiling"]

KSIG = (0.0, 0.26, 0.52, 0.78, 1.04, 1.30, 1.56, 2.08)
KT = (0.0, 0.035, 0.104)
ALPHAS = (0.01, 0.02, 0.05, 0.1, 0.2)

# (field, low, high). Ranges as in stage 5, plus three constants stage 5 did
# not sweep. Battery and panel are re-sized from the overridden energy model,
# so each cell keeps its autonomy and harvest ratio: what changes is the cost
# structure inside the regime, not the regime.
SENSITIVITY = [
    ("node.tier_b_wake_j", 3.0, 30.0),
    ("node.tier_b_active_w", 1.5, 6.0),
    ("node.e_capture_day_j", 0.2, 0.8),
    ("node.e_capture_night_j", 0.4, 3.0),
    ("node.p_sleep_w", 0.0005, 0.01),
    ("node.tier_a_pj_per_mac", 5.0, 100.0),
    ("solar.enclosure_gain_c", 0.0, 30.0),
    ("battery.soc_floor", 0.02, 0.15),
    ("battery.self_discharge_per_month", 0.01, 0.05),
    ("battery.eta_charge", 0.90, 0.99),
    ("control.deadline_h", 12.0, 96.0),
    ("control.min_ceiling", 0.15, 0.5),
]
WAKE_J = (0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 20.0, 30.0, 50.0)
WAKE_AUTONOMY = (0.5, 1.0, 2.0, 5.0, 30.0)
WAKE_RATIOS = (1.0, 2.0)
DEADLINES_H = (3.0, 6.0, 12.0, 24.0, 48.0, 96.0)
QUEUE_FRAMES = (25, 100, 400, 20000)
LATENCY_CELLS = ((0.5, 2.0), (1.0, 1.0), (2.0, 1.0))
# PREREGISTRATION.md, "Experimental grid" (fixed 2026-09-28, before any
# Serengeti data was downloaded).
PREREG_AUTONOMY = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0)
PREREG_RATIOS = (0.75, 1.0, 1.5, 2.0, 3.0)
PREREG_POLICIES = ["always_now", "ee_now", "ceiling_now", "lazy_defer", "defer_ceiling",
                   "batch_anytime", "ceiling_conformal"]


def cell_jobs(cfg, policies, cells, overrides=None, tags=None, site=None):
    jobs = []
    for a, r in cells:
        t = dict(autonomy=a, **(tags or {}))
        jobs += make_jobs(cfg, policies, [r], site=site, overrides=overrides,
                          autonomy_days=a, tags=t)
    return jobs


def coverage(cfg):
    """The reviewer asked whether 94.4% coverage against a 95% target is a
    calibration failure, and how the conformal bound compares with simpler
    reserve rules. The required reserve depends only on the capture stream and
    the weather, not on the battery or the policy, so one policy that queries
    the bound (sunsched) at the paper's 10 Wh sizing is enough: 9 cameras x 5
    years x 3 ratios = 135 distinct traces."""
    return make_jobs(cfg, ["sunsched"], RATIOS, tags=dict(group="coverage"))


def grid(cfg):
    """The reviewer's first request was a controller that works in both
    regimes. defer_ceiling is the simplest such controller (the two winning
    rules combined, defaults untouched); batch_anytime batches wakes without
    waiting for the sun, which separates wake amortisation from time-shifting;
    the *_conformal variants put the conformal bound in charge of the ceiling.
    The paper's own policies are re-run so every comparison is paired and the
    new ageing metrics exist for them."""
    cells = [(a, r) for a in AUTONOMY for r in RATIOS]
    return cell_jobs(cfg, PAPER_POLICIES + NEW_POLICIES, cells, tags=dict(group="grid"))


def sens(cfg):
    """The reviewer doubted the energy model. Each constant is pushed to both
    ends of its range at each claim cell; a claim that flips is reported."""
    jobs = []
    for field, lo, hi in SENSITIVITY:
        for level, value in (("low", lo), ("high", hi)):
            jobs += cell_jobs(cfg, CORE, CLAIM_CELLS, overrides={field: value},
                              tags=dict(group="sens", param=field, level=level, value=value))
    return jobs


def wake(cfg):
    """Stage 5 showed the deferral result can flip with the wake energy. This
    traces where: the break-even wake energy per autonomy level."""
    cells = [(a, r) for a in WAKE_AUTONOMY for r in WAKE_RATIOS]
    jobs = []
    for w in WAKE_J:
        jobs += cell_jobs(cfg, ["always_now", "lazy_defer", "batch_anytime"], cells,
                          overrides={"node.tier_b_wake_j": w},
                          tags=dict(group="wake", param="node.tier_b_wake_j", value=w))
    return jobs


def latency(cfg):
    """Deferral trades delay for value. The reviewer asked how the result
    depends on the deadline and on flash capacity for the queue."""
    jobs = []
    for d in DEADLINES_H:
        jobs += cell_jobs(cfg, ["lazy_defer", "defer_ceiling"], LATENCY_CELLS,
                          overrides={"control.deadline_h": d},
                          tags=dict(group="latency", param="control.deadline_h", value=d))
    for q in QUEUE_FRAMES:
        jobs += cell_jobs(cfg, ["lazy_defer", "defer_ceiling"], LATENCY_CELLS,
                          overrides={"node.queue_capacity": q},
                          tags=dict(group="latency", param="node.queue_capacity", value=q))
    return jobs


def sites(cfg):
    """Same real animal activity under other suns (climate transplants, not
    real deployments; labelled as such in the paper)."""
    jobs = []
    for site in SITES:
        if site in (cfg.experiment.main_site, "serengeti", "cct_region"):
            continue
        jobs += cell_jobs(cfg, CORE, CLAIM_CELLS, site=site,
                          tags=dict(group="sites", param="site", level=site))
    return jobs


def prereg(cfg):
    """The pre-registered grid. H2 and H3 of PREREGISTRATION.md are decided
    on its cells; the extra policies are reported as exploratory."""
    cells = [(a, r) for a in PREREG_AUTONOMY for r in PREREG_RATIOS]
    return cell_jobs(cfg, PREREG_POLICIES, cells, tags=dict(group="prereg"))


V3 = ["sunsched3", "sunsched3_no_coalesce", "sunsched3_no_pv", "sunsched3_no_hold"]
V3_DEV2 = ["sunsched3", "sunsched3_no_coalesce", "sunsched3_no_pv", "sunsched3_no_hold"]   # also iteration 3


def v3grid(cfg):
    """Development run of SunSched v3 and its ablations on the paper grid. The
    baselines are deterministic and already in grid_runs.csv, so only v3 runs."""
    cells = [(a, r) for a in AUTONOMY for r in RATIOS]
    return cell_jobs(cfg, V3, cells, tags=dict(group="v3grid"))


def v3wake(cfg):
    """v3 under the wake-energy sweep (baselines already in wake_runs.csv)."""
    cells = [(a, r) for a in WAKE_AUTONOMY for r in WAKE_RATIOS]
    jobs = []
    for w in WAKE_J:
        jobs += cell_jobs(cfg, ["sunsched3"], cells, overrides={"node.tier_b_wake_j": w},
                          tags=dict(group="v3wake", param="node.tier_b_wake_j", value=w))
    return jobs


def v3grid2(cfg):
    """Development iteration 2 of v3 on the paper grid (CCT20 only)."""
    cells = [(a, r) for a in AUTONOMY for r in RATIOS]
    return cell_jobs(cfg, V3_DEV2, cells, tags=dict(group="v3grid2"))


V3_DEV4 = ["sunsched3", "sunsched3_fc", "sunsched3_no_coalesce", "sunsched3_no_pv", "sunsched3_no_hold"]


def v3grid4(cfg):
    """Development iteration 4 of v3 on the paper grid (CCT20 only): the
    reservoir hold with both spending rules, and the ablations."""
    cells = [(a, r) for a in AUTONOMY for r in RATIOS]
    return cell_jobs(cfg, V3_DEV4, cells, tags=dict(group="v3grid4"))


def v3wake4(cfg):
    """Iteration 4 under the wake-energy sweep, both spending rules."""
    cells = [(a, r) for a in WAKE_AUTONOMY for r in WAKE_RATIOS]
    jobs = []
    for w in WAKE_J:
        jobs += cell_jobs(cfg, ["sunsched3", "sunsched3_fc"], cells, overrides={"node.tier_b_wake_j": w},
                          tags=dict(group="v3wake4", param="node.tier_b_wake_j", value=w))
    return jobs


# PREREGISTRATION_v2.md, "Experimental grid" (fixed before either test corpus
# was downloaded).
PREREG2_POLICIES = ["always_now", "ee_now", "ceiling_now", "lazy_defer",
                    "batch_anytime", "defer_ceiling", "ceiling_conformal", "defer_ceiling_conformal",
                    "sunsched3", "sunsched3_no_coalesce", "sunsched3_no_pv", "sunsched3_no_hold"]
PREREG2_WAKE_J = (1.0, 2.0, 3.0, 6.0)


def prereg2(cfg):
    """The grid of PREREGISTRATION_v2.md (HV1-HV4)."""
    cells = [(a, r) for a in PREREG_AUTONOMY for r in PREREG_RATIOS]
    return cell_jobs(cfg, PREREG2_POLICIES, cells, tags=dict(group="prereg2"))


def wake2(cfg):
    """The wake sweep of PREREGISTRATION_v2.md (HV5)."""
    cells = [(a, r) for a in WAKE_AUTONOMY for r in WAKE_RATIOS]
    jobs = []
    for w in PREREG2_WAKE_J:
        jobs += cell_jobs(cfg, ["always_now", "lazy_defer", "sunsched3"], cells,
                          overrides={"node.tier_b_wake_j": w},
                          tags=dict(group="wake2", param="node.tier_b_wake_j", value=w))
    return jobs


# Development iteration 5 (CCT20 only), on the layout of the pre-registered test
# grid (PREREG_AUTONOMY x PREREG_RATIOS), so the development table matches the
# test design. The paper grid has ratios 0.5, 1 and 2 only, so the baselines
# are added at 0.75, 1.5 and 3.
V3_DEV5 = ["sunsched3_rule", "sunsched3_rise"]
V3_DEV5_REF = ["sunsched3", "sunsched3_no_hold"]
NEW_RATIOS5 = (0.75, 1.5, 3.0)


def v3grid5(cfg):
    """Iteration 5: the two hold-engage rules on the pre-registered grid layout."""
    cells = [(a, r) for a in PREREG_AUTONOMY for r in PREREG_RATIOS]
    return cell_jobs(cfg, V3_DEV5, cells, tags=dict(group="v3grid5"))


def v3ref5(cfg):
    """Iteration 4 and the no-hold ablation on the pre-registered grid layout."""
    cells = [(a, r) for a in PREREG_AUTONOMY for r in PREREG_RATIOS]
    return cell_jobs(cfg, V3_DEV5_REF, cells, tags=dict(group="v3ref5"))


def base5reg(cfg):
    """The registered baselines at the pre-registered ratios the paper grid lacks."""
    cells = [(a, r) for a in PREREG_AUTONOMY for r in NEW_RATIOS5]
    return cell_jobs(cfg, PAPER_POLICIES, cells, tags=dict(group="base5reg"))


def base5new(cfg):
    """The further comparators at the pre-registered ratios the paper grid lacks."""
    cells = [(a, r) for a in PREREG_AUTONOMY for r in NEW_RATIOS5]
    return cell_jobs(cfg, NEW_POLICIES, cells, tags=dict(group="base5new"))


def v3wake5(cfg):
    """Iteration 5 under the pre-registered wake energies (HV5)."""
    cells = [(a, r) for a in WAKE_AUTONOMY for r in WAKE_RATIOS]
    jobs = []
    for w in PREREG2_WAKE_J:
        jobs += cell_jobs(cfg, V3_DEV5, cells, overrides={"node.tier_b_wake_j": w},
                          tags=dict(group="v3wake5", param="node.tier_b_wake_j", value=w))
    return jobs


def v3frozen(cfg):
    """The frozen v3 and its two remaining ablations on the pre-registered grid
    layout (CCT20). The frozen sunsched3 must reproduce sunsched3_rise of
    v3grid5 exactly; the ablations are exploratory."""
    cells = [(a, r) for a in PREREG_AUTONOMY for r in PREREG_RATIOS]
    return cell_jobs(cfg, ["sunsched3", "sunsched3_no_coalesce", "sunsched3_no_pv"], cells,
                     tags=dict(group="v3frozen"))


# -- v4 development screening (cct20, cct_new, serengeti S10 and S11 are all seen data) --
V4_SCREEN_REF = ["always_now", "lazy_defer", "ee_now", "ceiling_now", "batch_anytime",
                 "defer_ceiling_conformal", "sunsched3", "sunsched3_no_hold"]
V4_SCREEN_YEARS = (2011, 2013)


def v4screen(cfg):
    from sunsched.experiment import V4_POLICIES
    cells = [(a, r) for a in PREREG_AUTONOMY for r in PREREG_RATIOS]
    cells_jobs = cell_jobs(cfg, V4_SCREEN_REF + V4_POLICIES, cells, tags=dict(group="v4screen"))
    return [j for j in cells_jobs if int(j["year"]) in V4_SCREEN_YEARS]


def v5screen(cfg):
    """PLAN_V2 gate D6 (first look, frozen-trunk outcomes): one warm-up year, two weather years."""
    from sunsched.experiment import LAZY_CEILINGS, V5_POLICIES
    pol = ["always_now", "lazy_defer", "sunsched3_no_hold", "v4_full_surv"] + V5_POLICIES + list(LAZY_CEILINGS)
    cells = [(a, r) for a in (0.5, 1, 2, 5, 10, 30, 100) for r in (1.0, 1.5, 2.0)]
    jobs = cell_jobs(cfg, pol, cells, tags=dict(group="v5screen"))
    return [dict(j, warmup_days=365) for j in jobs if int(j["year"]) in V4_SCREEN_YEARS]


def v5capscreen(cfg):
    """The v5 front with a cap, on the cells and years of v5screen."""
    from sunsched.experiment import V5_CAPS
    cells = [(a, r) for a in (0.5, 1, 2, 5, 10, 30, 100) for r in (1.0, 1.5, 2.0)]
    jobs = cell_jobs(cfg, list(V5_CAPS), cells, tags=dict(group="v5capscreen"))
    return [dict(j, warmup_days=365) for j in jobs if int(j["year"]) in V4_SCREEN_YEARS]


def v6screen(cfg):
    """The buffer rule, on the cells and years of v5screen."""
    from sunsched.experiment import V6_VARIANTS
    cells = [(a, r) for a in (0.5, 1, 2, 5, 10, 30, 100) for r in (1.0, 1.5, 2.0)]
    jobs = cell_jobs(cfg, list(V6_VARIANTS), cells, tags=dict(group="v6screen"))
    return [dict(j, warmup_days=365) for j in jobs if int(j["year"]) in V4_SCREEN_YEARS]


QTEST_CELLS = [(a, r) for a in (0.5, 1, 2) for r in (1.0, 1.5, 2.0)] +               [(a, r) for a in (5, 10) for r in (1.0, 1.5, 2.0)] + [(a, r) for a in (30, 100) for r in (1.5, 2.0)]
QTEST_YEARS = (2019, 2021, 2023)


def qtest(cfg):
    """PREREGISTRATION_v3.md: the frozen test on a new corpus, run once."""
    from sunsched.experiment import LAZY_CEILINGS, QTEST_FRONT
    pol = ["always_now", "ceiling_now", "lazy_defer", "sunsched3"] + list(LAZY_CEILINGS) + list(QTEST_FRONT)
    jobs = cell_jobs(cfg, pol, QTEST_CELLS, tags=dict(group="qtest"))
    return [dict(j, warmup_days=365) for j in jobs if int(j["year"]) in QTEST_YEARS]


QNODE = (("wake3", "node.tier_b_wake_j", 3.0), ("wake35", "node.tier_b_wake_j", 35.0),
         ("sleep05", "node.p_sleep_w", 0.0005), ("deadline12", "control.deadline_h", 12.0))


def qnode(cfg):
    """Other nodes and another deadline: does W, recomputed for each, still predict the gain?"""
    year = 2021 if cfg.data.corpus.startswith("lila_") else 2013
    jobs = []
    for name, field, value in QNODE:
        jobs += cell_jobs(cfg, ["always_now", "lazy_defer"], [(a, r) for a in (0.5, 1, 2) for r in (1.0, 1.5, 2.0)],
                          overrides={field: value}, tags=dict(group="qnode", node=name))
    return [dict(j, warmup_days=365) for j in jobs if int(j["year"]) == year]


def qsmoke(cfg):
    """A few jobs of qtest, to check the pipeline on a corpus that is not a test corpus."""
    jobs = [j for j in qtest(cfg) if int(j["year"]) == 2021 and float(j["tag_autonomy"]) in (1.0, 30.0) and float(j["ratio"]) == 1.5]
    keep = sorted({j["location"] for j in jobs})[:3]
    return [j for j in jobs if j["location"] in keep]


GROUPS = dict(qnode=qnode, qsmoke=qsmoke, qtest=qtest, v6screen=v6screen, v5capscreen=v5capscreen, v5screen=v5screen, v4screen=v4screen, coverage=coverage, grid=grid, sens=sens, wake=wake, latency=latency, sites=sites,
              prereg=prereg, v3grid=v3grid, v3wake=v3wake, v3grid2=v3grid2, v3grid4=v3grid4, v3wake4=v3wake4,
              prereg2=prereg2, wake2=wake2, v3grid5=v3grid5, v3ref5=v3ref5, base5reg=base5reg,
              base5new=base5new, v3wake5=v3wake5, v3frozen=v3frozen)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--only", nargs="*", default=["coverage", "grid", "sens", "wake", "latency", "sites"])
    ap.add_argument("--dry-run", action="store_true", help="print job counts and exit")
    args = ap.parse_args()
    cfg = load_cfg(args)
    if not os.path.exists(f"{cfg.artifacts_dir}/outcomes.npz"):
        sys.exit(f"{cfg.artifacts_dir}/outcomes.npz not found. Point --out at the folder of the "
                 f"earlier full run (the one holding artifacts/), or run stages 1-3 first.")
    if not set(args.only) <= {"qtest", "qsmoke", "qnode"}:            # variants that the frozen test does not score
        cfg.aging.variant_k_sigmas = KSIG
        cfg.aging.variant_k_Ts = KT
        cfg.control.shadow_alphas = ALPHAS

    out = f"{cfg.results_dir}/tables/revision"
    os.makedirs(out, exist_ok=True)
    plan = {name: GROUPS[name](cfg) for name in args.only}
    total = sum(len(j) for j in plan.values())
    for name, jobs in plan.items():
        print(f"  {name:<9} {len(jobs):>6} simulated deployments")
    print(f"  {'total':<9} {total:>6}")
    if args.dry_run:
        return
    with open(f"{out}/manifest_{'_'.join(args.only)}.json", "w", encoding="utf-8") as f:
        json.dump(dict(started=datetime.now().isoformat(timespec="seconds"), git=git_commit(),
                       corpus=cfg.data.corpus, backbone=cfg.vision.backbone,
                       groups={k: len(v) for k, v in plan.items()},
                       config=config_to_dict(cfg)), f, indent=2, default=str)
    for name, jobs in plan.items():
        banner(f"revision: {name} ({len(jobs)} runs)")
        rows = run_jobs(jobs, cfg.experiment.n_workers, name)
        write_csv(f"{out}/{name}_runs.csv", rows)
        bad = sum("error" in r for r in rows)
        print(f"  wrote {out}/{name}_runs.csv ({len(rows)} rows, {bad} errors)")


if __name__ == "__main__":
    main()
