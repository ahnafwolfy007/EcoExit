#!/usr/bin/env python
"""The pre-registered hypotheses of PREREGISTRATION.md, tested exactly as written.

    python pipeline/06_hypotheses.py --corpus serengeti     # the test that counts
    python pipeline/06_hypotheses.py --corpus cct20         # development: informative only

H1  mismatch replicates: >= 40% of animal captures with the sun down, negative
    hourly correlation between captures and solar energy.
H2  small storage: lazy_defer beats always_now on value by >= 0.02 in >= 5 of
    9 cells (autonomy 0.5/1/2 d x ratio 1.0/1.5/2.0), paired Wilcoxon,
    Holm-corrected.
H3  large storage: ceiling_now lives >= 1.15x longer than always_now at value
    >= -0.01 in >= 4 of 6 cells (autonomy 30/100 d x ratio 1.0/1.5/2.0).
H4  as H2, with sunsched in place of lazy_defer.
H5  sunsched is near-best (value within 0.01 of the best baseline, life
    >= 0.95x the longest-lived near-best baseline) in >= 75% of cells with
    ratio >= 1.0.

H1-H3 are the primary claim. Exit code 0 if they all hold, 2 otherwise; a 2 is
a result, not a crash. Only the serengeti run decides anything; a cct20 run
reports the same numbers for development.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sunsched.cli import banner, base_parser, load_cfg, read_csv, write_csv, write_json
from sunsched.eval.stats import holm, paired_wilcoxon


def paired(rows, autonomy, ratio, policy, metric):
    out = {}
    for r in rows:
        if (r["policy"] == policy and math.isclose(r["autonomy"], autonomy)
                and math.isclose(r["ratio"], ratio)):
            v = float(r[metric])
            if np.isfinite(v):
                out[(r["location"], r["year"])] = v
    return out


def cell_value_test(rows, cells, ours, base, h):
    """Per cell: mean paired value difference and Wilcoxon p; Holm over cells."""
    res = []
    for a, r in cells:
        vo, vb = paired(rows, a, r, ours, "value_score"), paired(rows, a, r, base, "value_score")
        keys = sorted(set(vo) & set(vb))
        t = paired_wilcoxon([vo[k] for k in keys], [vb[k] for k in keys])
        res.append(dict(autonomy=a, ratio=r, policy=ours, versus=base, n=t["n"],
                        value_diff=t["mean_diff"], p_value=t["p_value"]))
    for x, adj in zip(res, holm([x["p_value"] for x in res], h.alpha)):
        x["p_holm"] = adj["p_adjusted"]
        x["holds"] = bool(np.isfinite(x["value_diff"]) and x["value_diff"] >= h.value_margin
                          and np.isfinite(x["p_holm"]) and x["p_holm"] < h.alpha)
    return res


def cell_life_test(rows, cells, ours, base, h):
    res = []
    for a, r in cells:
        lo, lb = (paired(rows, a, r, ours, "battery_years_to_eol"),
                  paired(rows, a, r, base, "battery_years_to_eol"))
        vo, vb = paired(rows, a, r, ours, "value_score"), paired(rows, a, r, base, "value_score")
        keys = sorted(set(lo) & set(lb) & set(vo) & set(vb))
        la = np.log([max(lo[k], 1e-9) for k in keys])
        lbb = np.log([max(lb[k], 1e-9) for k in keys])
        t = paired_wilcoxon(la, lbb)
        res.append(dict(autonomy=a, ratio=r, policy=ours, versus=base, n=len(keys),
                        life_ratio=float(np.exp(np.mean(la - lbb))) if keys else float("nan"),
                        value_diff=float(np.mean([vo[k] - vb[k] for k in keys])) if keys else float("nan"),
                        p_value=t["p_value"]))
    for x, adj in zip(res, holm([x["p_value"] for x in res], h.alpha)):
        x["p_holm"] = adj["p_adjusted"]
        x["holds"] = bool(np.isfinite(x["life_ratio"]) and x["life_ratio"] >= h.life_ratio
                          and x["value_diff"] >= -h.value_tolerance
                          and np.isfinite(x["p_holm"]) and x["p_holm"] < h.alpha)
    return res


def near_best(rows, h, ratios_min=1.0):
    cells = sorted({(r["autonomy"], r["ratio"]) for r in rows
                    if np.isfinite(r["autonomy"]) and r["ratio"] >= ratios_min - 1e-9})
    out = []
    for a, r in cells:
        means = {}
        for p in list(h.h5_baselines) + ["sunsched"]:
            v = list(paired(rows, a, r, p, "value_score").values())
            l = list(paired(rows, a, r, p, "battery_years_to_eol").values())
            if v and l:
                means[p] = (float(np.mean(v)), float(np.mean(l)))
        if "sunsched" not in means or not any(p in means for p in h.h5_baselines):
            continue
        base = {p: m for p, m in means.items() if p in h.h5_baselines}
        V = max(v for v, _ in base.values())
        L = max(l for v, l in base.values() if v >= V - h.value_tolerance)
        sv, sl = means["sunsched"]
        out.append(dict(autonomy=a, ratio=r, best_value=V, best_life=L, sunsched_value=sv,
                        sunsched_life=sl, holds=bool(sv >= V - h.value_tolerance
                                                     and sl >= (1 - h.life_tolerance) * L)))
    return out


def required(min_cells, total, present, quick):
    if not quick:
        return min_cells
    return max(1, math.ceil(min_cells / total * present))


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg = load_cfg(args)
    h = cfg.hypothesis
    tables = f"{cfg.results_dir}/tables"
    rows = [r for r in read_csv(f"{tables}/runs.csv") if not (isinstance(r.get("error"), str) and r["error"])]
    present = {(r["autonomy"], r["ratio"]) for r in rows}

    def cells_of(autonomies):
        wanted = [(a, r) for a in autonomies for r in h.test_ratios]
        have = [c for c in wanted if any(math.isclose(c[0], p[0]) and math.isclose(c[1], p[1]) for p in present)]
        if len(have) < len(wanted) and not args.quick:
            raise SystemExit(f"runs.csv lacks cells {sorted(set(wanted) - set(have))}; run stage 4 in full")
        return wanted if not args.quick else have

    small, large = cells_of(h.small_autonomy), cells_of(h.large_autonomy)
    n_small_all = len(h.small_autonomy) * len(h.test_ratios)
    n_large_all = len(h.large_autonomy) * len(h.test_ratios)

    banner(f"PRE-REGISTERED HYPOTHESES  ({cfg.data.corpus}: "
           f"{'TEST -- this run decides' if cfg.data.corpus == 'serengeti' else 'development, informative only'})")
    if args.quick:
        print("  quick preset: cells and replicates are reduced; these verdicts are not results.")

    env = json.load(open(f"{tables}/environment_summary.json", encoding="utf-8"))
    h1 = bool(env["events_sun_below_horizon"] >= h.h1_min_night_share
              and env["pearson_hourly_events_vs_energy"] < 0)
    print(f"\n  H1  {env['events_sun_below_horizon'] * 100:.1f}% of {env['n_animal_captures']:,} animal "
          f"captures with the sun down (needs >= {h.h1_min_night_share * 100:.0f}%), hourly r = "
          f"{env['pearson_hourly_events_vs_energy']:.2f} (needs < 0)  -> {'HOLDS' if h1 else 'fails'}")

    def report(name, res, need, what):
        n = sum(x["holds"] for x in res)
        ok = n >= need and len(res) > 0
        print(f"\n  {name}  {what}: holds in {n}/{len(res)} cells (needs {need})  -> {'HOLDS' if ok else 'fails'}")
        for x in res:
            extra = (f"life x{x['life_ratio']:.2f}, " if "life_ratio" in x else "")
            print(f"      autonomy {x['autonomy']:>5} d, ratio {x['ratio']:>4}: {extra}"
                  f"value {x['value_diff']:+.3f}, p_holm {x['p_holm']:.3g}, n={x['n']}  "
                  f"{'yes' if x['holds'] else 'no'}")
        return ok

    h2_res = cell_value_test(rows, small, "lazy_defer", "always_now", h)
    h2 = report("H2", h2_res, required(h.h2_min_cells, n_small_all, len(small), args.quick),
                "lazy_defer beats always_now at small storage")
    h3_res = cell_life_test(rows, large, "ceiling_now", "always_now", h)
    h3 = report("H3", h3_res, required(h.h3_min_cells, n_large_all, len(large), args.quick),
                "ceiling_now outlives always_now at large storage")
    h4_res = cell_value_test(rows, small, "sunsched", "always_now", h)
    h4 = report("H4", h4_res, required(h.h2_min_cells, n_small_all, len(small), args.quick),
                "sunsched beats always_now at small storage")
    h5_res = near_best(rows, h)
    share = float(np.mean([x["holds"] for x in h5_res])) if h5_res else 0.0
    h5 = bool(h5_res) and share >= h.h5_min_share
    print(f"\n  H5  sunsched near-best in {sum(x['holds'] for x in h5_res)}/{len(h5_res)} cells "
          f"({share * 100:.0f}%, needs {h.h5_min_share * 100:.0f}%)  -> {'HOLDS' if h5 else 'fails'}")
    for x in h5_res:
        if not x["holds"]:
            print(f"      autonomy {x['autonomy']:>5} d, ratio {x['ratio']:>4}: value {x['sunsched_value']:.3f} "
                  f"vs best {x['best_value']:.3f}, life {x['sunsched_life']:.1f} vs {x['best_life']:.1f}")

    primary = h1 and h2 and h3
    banner(f"PRIMARY CLAIM (H1-H3): {'HOLDS' if primary else 'DOES NOT HOLD'}")
    claims = {(True, True, True): "The storage-dependent regime map holds on held-out data.",
              (True, True, False): "Deferral pays at small storage; the charge-ceiling result does not generalise.",
              (True, False, True): "Charge ceilings pay at large storage; deferral does not generalise.",
              (True, False, False): "Only the measurement (H1) can be claimed."}
    msg = claims.get((h1, h2, h3), "The day/night mismatch did not replicate on this corpus.")
    print(f"  {msg}")
    print(f"  Method: SunSched v2 {'beats' if h4 else 'does not beat'} always_now at small storage (H4); "
          f"{'is' if h5 else 'is not'} near-best across the grid (H5).")

    write_csv(f"{tables}/hypotheses_cells.csv",
              [dict(hypothesis=k, **x) for k, res in (("H2", h2_res), ("H3", h3_res), ("H4", h4_res),
                                                      ("H5", h5_res)) for x in res])
    write_json(f"{tables}/hypotheses.json", dict(
        corpus=cfg.data.corpus, decides=cfg.data.corpus == "serengeti" and not args.quick,
        quick=bool(args.quick), H1=h1, H2=h2, H3=h3, H4=h4, H5=h5, primary=primary, claim=msg,
        h5_share=share, margins=dict(value_margin=h.value_margin, value_tolerance=h.value_tolerance,
                                     life_ratio=h.life_ratio, life_tolerance=h.life_tolerance,
                                     alpha=h.alpha)))
    return 0 if primary else 2


if __name__ == "__main__":
    raise SystemExit(main())
