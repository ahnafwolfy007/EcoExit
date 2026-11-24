"""Scores PREREGISTRATION_v3.md on one or more corpora.

    py analysis/score_v3.py --runs lila_swg=A_runs.csv lila_orinoquia=B_runs.csv ... --policy v5 --out DIR

Q1  predictor: Spearman (cameras of all corpora) between the avoidable wake share
    W and the small-storage gain of lazy deferral; per-corpus signs; gain ratio of
    W >= 0.05 to W < 0.02 cameras.
Q2  deferral: pooled small-storage gain of the registered policy over Immediate,
    per corpus, for cameras with W >= 0.05 and for all cameras.
Q3  life: geometric-mean life ratio and value change at large storage.
Q4  front: hypervolume of the registered front against lazy deferral with static
    ceilings.
Camera is the unit everywhere: each camera's effect is averaged over the cells
and weather years first.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sunsched.eval.pareto import hypervolume, normalise                                     # noqa: E402

SMALL = [(a, r) for a in (0.5, 1.0, 2.0) for r in (1.0, 1.5, 2.0)]
LARGE = [(a, r) for a in (30.0, 100.0) for r in (1.5, 2.0)]
LAZY_FRONT = ["lazy_defer", "lazy_c40", "lazy_c50", "lazy_c60", "lazy_c70", "lazy_c85"]
K = ["autonomy", "ratio", "location", "year"]


def load(path):
    d = pd.read_csv(path, low_memory=False)
    bad = d["error"].notna() & (d["error"].astype(str) != "") if "error" in d.columns else pd.Series(False, index=d.index)
    d = d[~bad].rename(columns={"tag_autonomy": "autonomy"})
    d["location"] = d.location.astype(str)
    return d, int(bad.sum())


def in_cells(d, cells):
    return d[[c in cells for c in zip(d.autonomy.astype(float), d.ratio.astype(float))]]


def per_camera(d, a, b, metric, cells, log=False):
    x = in_cells(d, set(cells)).pivot_table(index=K, columns="policy", values=metric)
    v = np.log(x[a] / x[b]) if log else x[a] - x[b]
    return v.groupby("location").mean()


def greater(x):
    x = np.asarray(x)
    return float(wilcoxon(x, alternative="greater").pvalue) if len(x) >= 5 and not np.allclose(x, 0) else float("nan")


def boot(x, n=4000, seed=0):
    r, x = np.random.default_rng(seed), np.asarray(x)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def hv_per_camera(d, policies, cells):
    """Mean over cells and years of each camera's hypervolume."""
    s = in_cells(d, set(cells))
    s = s[s.policy.isin(policies)]
    x, y = normalise(s.value_score, s.ref_value_full, s.battery_years_to_eol)
    s = s.assign(x=x, y=y)
    return s.groupby(K)[["x", "y"]].apply(lambda g: hypervolume(g.values)).groupby("location").mean()


def score(runs, policy, front):
    out, cams = {}, []
    for name, d in runs.items():
        W = d.groupby("location").wake_share.first()
        gl = per_camera(d, "lazy_defer", "always_now", "value_score", SMALL)
        gp = per_camera(d, policy, "always_now", "value_score", SMALL)
        life = per_camera(d, policy, "always_now", "battery_years_to_eol", LARGE, log=True)
        dv = per_camera(d, policy, "always_now", "value_score", LARGE)
        n = d.groupby("location").n_frames.first()
        cams.append(pd.DataFrame(dict(corpus=name, W=W, gain_lazy=gl, gain_policy=gp, log_life=life, dv_large=dv, frames=n)).reset_index())
        r = dict(cameras=int(len(W)), frames=int(n.sum()), W_median=float(W.median()),
                 Q2_gain_all=float(gp.mean()), Q2_ci=boot(gp), Q2_p=greater(gp), Q2_gain_lazy=float(gl.mean()),
                 Q3_life=float(np.exp(life.mean())), Q3_p=greater(life), Q3_dv=float(dv.mean()), Q3_dv_ci=boot(dv),
                 within_rho=float(spearmanr(W.loc[gl.index], gl)[0]))
        if set(front) <= set(d.policy) and set(LAZY_FRONT) <= set(d.policy):
            cells = sorted(set(zip(d.autonomy.astype(float), d.ratio.astype(float))))
            a, b = hv_per_camera(d, front, cells), hv_per_camera(d, LAZY_FRONT, cells)
            r.update(Q4_hv=float(a.mean()), Q4_hv_lazy=float(b.mean()), Q4_rel=float(a.mean() / b.mean() - 1), Q4_p=greater(a - b.loc[a.index]))
        out[name] = r
    c = pd.concat(cams, ignore_index=True)
    rho, p2 = spearmanr(c.W, c.gain_lazy)
    hi, lo = c[c.W >= 0.05], c[c.W < 0.02]
    out["pooled"] = dict(cameras=int(len(c)), Q1_rho=float(rho), Q1_p_one_sided=float(p2 / 2 if rho > 0 else 1 - p2 / 2),
                         Q1_rho_frames=float(spearmanr(c.frames, c.gain_lazy)[0]),
                         Q1_within_positive=int(sum(out[k]["within_rho"] > 0 for k in runs)),
                         n_hi=int(len(hi)), n_lo=int(len(lo)), gain_hi=float(hi.gain_lazy.mean()), gain_lo=float(lo.gain_lazy.mean()),
                         gain_policy_hi=float(hi.gain_policy.mean()), gain_policy_lo=float(lo.gain_policy.mean()),
                         gain_policy_hi_p=greater(hi.gain_policy) if len(hi) else float("nan"))
    return out, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--policy", default="v5")
    ap.add_argument("--front", nargs="+", default=["v5_m00", "v5", "v5_m25", "v5_m50", "v5_nohold"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = {}
    for x in a.runs:
        name, path = x.split("=", 1)
        runs[name], bad = load(path)
        print(name, len(runs[name]), "runs,", bad, "errors")
    out, cams = score(runs, a.policy, a.front)
    cams.to_csv(f"{a.out}/cameras.csv", index=False)
    with open(f"{a.out}/score_v3.json", "w") as f:
        json.dump(out, f, indent=1)
    print(pd.DataFrame(out).T.round(4).T.to_string())


if __name__ == "__main__":
    main()
