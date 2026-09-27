#!/usr/bin/env python
"""The kill test: does SunSched beat the strongest existing approaches?

    python pipeline/06_kill_test.py

Pre-registered before any results existed, with margins set in
ExperimentCfg. For each regime, SunSched must beat each strong baseline
(per-frame energy-aware early exit, the same plus a charge ceiling, and lazy
deferral) on at least one of:

  A. value score higher by >= kill_value_margin, battery life no more than
     kill_life_tolerance worse;
  B. battery life at least kill_life_ratio times longer, value score no more
     than kill_value_tolerance lower;

with a paired Wilcoxon test over (camera, weather year) significant at 0.05
after Holm correction within the regime. The overall verdict is PASS if SunSched
beats all three baselines in at least half the regimes.

Exit code 0 = PASS, 2 = FAIL. A FAIL is a result: it means the idea, as
built, does not beat what already exists, and the paper should not claim it
does.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sunsched.cli import banner, base_parser, load_cfg, read_csv, write_csv, write_json
from sunsched.eval.stats import holm, paired_wilcoxon
from sunsched.experiment import STRONG_BASELINES


def paired(rows, ratio, policy, metric):
    return {(r["location"], r["year"]): float(r[metric]) for r in rows
            if r["ratio"] == ratio and r["policy"] == policy and np.isfinite(float(r[metric]))}


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg = load_cfg(args)
    e = cfg.experiment
    rows = [r for r in read_csv(f"{cfg.results_dir}/tables/runs.csv")
            if not (isinstance(r.get("error"), str) and r.get("error"))]

    banner("KILL TEST")
    print(f"  A: value +{e.kill_value_margin} at battery life >= {1 - e.kill_life_tolerance:.2f}x")
    print(f"  B: battery life >= {e.kill_life_ratio}x at value >= -{e.kill_value_tolerance}")
    table, per_ratio = [], {}
    for ratio in sorted({r["ratio"] for r in rows}):
        tests = []
        for base in STRONG_BASELINES:
            vs, vb = paired(rows, ratio, "sunsched", "value_score"), paired(rows, ratio, base, "value_score")
            ls, lb = (paired(rows, ratio, "sunsched", "battery_years_to_eol"),
                      paired(rows, ratio, base, "battery_years_to_eol"))
            keys = sorted(set(vs) & set(vb) & set(ls) & set(lb))
            if len(keys) < 6:
                print(f"  ratio {ratio}: only {len(keys)} paired replicates vs {base}; need >= 6")
                continue
            dv = np.array([vs[k] - vb[k] for k in keys])
            lr = np.array([ls[k] / max(lb[k], 1e-9) for k in keys])
            t_val = paired_wilcoxon([vs[k] for k in keys], [vb[k] for k in keys])
            t_life = paired_wilcoxon(np.log([ls[k] for k in keys]), np.log([max(lb[k], 1e-9) for k in keys]))
            tests.append(dict(ratio=ratio, versus=base, n=len(keys),
                              value_diff=float(dv.mean()), life_ratio=float(np.exp(np.log(lr).mean())),
                              p_value=t_val["p_value"], p_life=t_life["p_value"]))
        if not tests:
            continue
        adj_v = holm([t["p_value"] for t in tests] + [t["p_life"] for t in tests])
        for i, t in enumerate(tests):
            pv, pl = adj_v[i]["p_adjusted"], adj_v[len(tests) + i]["p_adjusted"]
            a = (t["value_diff"] >= e.kill_value_margin and t["life_ratio"] >= 1 - e.kill_life_tolerance
                 and pv < 0.05)
            b = (t["life_ratio"] >= e.kill_life_ratio and t["value_diff"] >= -e.kill_value_tolerance
                 and pl < 0.05)
            t.update(p_value_holm=pv, p_life_holm=pl, pass_A=bool(a), pass_B=bool(b), beats=bool(a or b))
            table.append(t)
            print(f"  ratio {ratio:<4} vs {t['versus']:<12} value {t['value_diff']:+.3f} (p={pv:.3g})  "
                  f"life x{t['life_ratio']:.2f} (p={pl:.3g})  -> {'BEATS' if t['beats'] else 'does not beat'}")
        per_ratio[ratio] = all(t["beats"] for t in tests) and len(tests) == len(STRONG_BASELINES)

    passed = bool(per_ratio) and sum(per_ratio.values()) >= max(1, (len(per_ratio) + 1) // 2)
    banner(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    for ratio, ok in per_ratio.items():
        print(f"  ratio {ratio}: {'beats all strong baselines' if ok else 'does not beat all'}")
    if not passed:
        print("\n  SunSched does not beat the strongest existing approaches by the")
        print("  pre-registered margins. Do not claim it does. Look at frontier.csv:")
        print("  if its curve still dominates somewhere, that narrower claim may hold.")
    write_csv(f"{cfg.results_dir}/tables/kill_test.csv", table)
    write_json(f"{cfg.results_dir}/tables/kill_test.json",
               dict(passed=passed, per_ratio={str(k): v for k, v in per_ratio.items()},
                    margins=dict(value_margin=e.kill_value_margin, life_ratio=e.kill_life_ratio,
                                 value_tolerance=e.kill_value_tolerance,
                                 life_tolerance=e.kill_life_tolerance),
                    baselines=STRONG_BASELINES))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
