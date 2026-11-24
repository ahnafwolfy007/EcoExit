"""Scores PREREGISTRATION_v3.md on the new corpora.

    py analysis/score_v3.py --runs lila_swg=A_runs.csv lila_orinoquia=B_runs.csv ...
                            --nodes lila_swg=A_node.csv lila_orinoquia=B_node.csv ... --out DIR

H1  predictor   Spearman between the avoidable wake share W and the small-storage
                gain of lazy deferral over Immediate.
H1n nodes       the same across other wake energies, sleep powers and deadlines, with W
                recomputed; the frame count cannot follow such a change.
H2  deferral    the registered controller against Immediate on cameras with W >= 0.05.
H3  life        battery life and value at large storage.
H4  front       hypervolume of the controller's value-life front against lazy
                deferral with static ceilings.
Camera is the unit everywhere: a camera's effect is averaged over cells and
weather years before any test. Pass rules are in `verdicts` and nowhere else.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sunsched.eval.pareto import hypervolume, normalise                                     # noqa: E402

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
SMALL = [(a, r) for a in (0.5, 1.0, 2.0) for r in (1.0, 1.5, 2.0)]
MID = [(a, r) for a in (5.0, 10.0) for r in (1.0, 1.5, 2.0)]
LARGE = [(a, r) for a in (30.0, 100.0) for r in (1.5, 2.0)]
LAZY_FRONT = ["lazy_defer", "lazy_c40", "lazy_c50", "lazy_c60", "lazy_c70", "lazy_c85"]
POLICY, FRONT = "v7", ["v7", "v7_c85", "v7_c70", "v7_c60", "v7_c50", "v7_c40"]
K = ["autonomy", "ratio", "location", "year"]


def load(path):
    d = pd.read_csv(path, low_memory=False)
    bad = d["error"].notna() & (d["error"].astype(str) != "") if "error" in d.columns else pd.Series(False, index=d.index)
    d = d[~bad].rename(columns={"tag_autonomy": "autonomy"})
    d["location"] = d.location.astype(str)
    d = d.rename(columns={"tag_node": "node"})
    keep = K + ["policy", "node", "value_score", "ref_value_full", "battery_years_to_eol", "wake_share", "n_frames", "dropped_frac",
                "b_wakes_per_day", "soc_start", "soc_end", "warmup_days", "days", "n_capture_slots", "n_stab", "demand_always_now_kj"]
    return d[[c for c in keep if c in d.columns]].copy(), int(bad.sum())


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
    if not len(x):
        return float("nan"), float("nan")
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def holm(ps):
    """Holm-adjusted p values, in the order given."""
    ps = np.asarray(ps, float)
    order, out, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out.tolist()


def hv_per_camera(d, policies, cells):
    """Mean over cells and years of each camera's hypervolume."""
    s = in_cells(d, set(cells))
    s = s[s.policy.isin(policies)].copy()
    s["x"], s["y"] = normalise(s.value_score, s.ref_value_full, s.battery_years_to_eol)
    return s.groupby(K)[["x", "y"]].apply(lambda g: hypervolume(g.values)).groupby("location").mean()


def score(runs, policy=POLICY, front=FRONT):
    out, cams = {}, []
    for name, d in runs.items():
        W = d.groupby("location").wake_share.first()
        n = d.groupby("location").n_frames.first()
        c = pd.DataFrame(dict(corpus=name, W=W, frames=n,
                              gain_lazy=per_camera(d, "lazy_defer", "always_now", "value_score", SMALL),
                              gain_policy=per_camera(d, policy, "always_now", "value_score", SMALL),
                              gain_policy_lazy=per_camera(d, policy, "lazy_defer", "value_score", SMALL),
                              log_life=per_camera(d, policy, "always_now", "battery_years_to_eol", LARGE, log=True),
                              dv_large=per_camera(d, policy, "always_now", "value_score", LARGE),
                              dv_mid=per_camera(d, policy, "always_now", "value_score", MID),
                              dv_mid_lazy=per_camera(d, policy, "lazy_defer", "value_score", MID)))
        r = dict(cameras=int(len(c)), frames=int(n.sum()), W_median=float(W.median()), W_min=float(W.min()), W_max=float(W.max()),
                 rho=float(spearmanr(c.W, c.gain_lazy)[0]), gain_lazy=float(c.gain_lazy.mean()), gain_policy=float(c.gain_policy.mean()),
                 gain_policy_ci=boot(c.gain_policy), gain_policy_p=greater(c.gain_policy), policy_minus_lazy_small=float(c.gain_policy_lazy.mean()),
                 dv_mid=float(c.dv_mid.mean()), dv_mid_lazy=float(c.dv_mid_lazy.mean()),
                 life=float(np.exp(c.log_life.mean())), life_ci=[float(np.exp(x)) for x in boot(c.log_life)], life_p=greater(c.log_life),
                 dv_large=float(c.dv_large.mean()), dv_large_ci=boot(c.dv_large))
        if set(front) <= set(d.policy) and set(LAZY_FRONT) <= set(d.policy):
            cells = sorted(set(zip(d.autonomy.astype(float), d.ratio.astype(float))))
            for lab, cc in (("all", cells), ("small", SMALL), ("mid", MID), ("large", LARGE)):
                a, b = hv_per_camera(d, front, cc), hv_per_camera(d, LAZY_FRONT, cc)
                b = b.loc[a.index]
                r.update({f"hv_{lab}": float(a.mean()), f"hv_{lab}_lazy": float(b.mean()), f"hv_{lab}_rel": float(a.mean() / b.mean() - 1),
                          f"hv_{lab}_p": greater(a - b), f"hv_{lab}_better": int((a > b).sum())})
                c[f"hv_{lab}"], c[f"hv_{lab}_lazy"] = a, b
        cams.append(c.reset_index())
        out[name] = r
    c = pd.concat(cams, ignore_index=True)
    rho, p2 = spearmanr(c.W, c.gain_lazy)
    hi, lo = c[c.W >= 0.05], c[c.W < 0.02]
    out["pooled"] = dict(cameras=int(len(c)), rho=float(rho), rho_p=float(p2 / 2 if rho > 0 else 1 - p2 / 2),
                         rho_frames=float(spearmanr(c.frames, c.gain_lazy)[0]), n_hi=int(len(hi)), n_lo=int(len(lo)),
                         gain_lazy_hi=float(hi.gain_lazy.mean()), gain_lazy_lo=float(lo.gain_lazy.mean()),
                         gain_policy_hi=float(hi.gain_policy.mean()), gain_policy_hi_ci=boot(hi.gain_policy), gain_policy_hi_p=greater(hi.gain_policy),
                         gain_policy_lo=float(lo.gain_policy.mean()))
    return out, c


def node_table(runs, nodes):
    """One row per camera and node: W, frames and the small-storage gain of lazy deferral. The registered
    node comes from the main runs, in the weather year of the node runs."""
    rows = []
    for name, n in nodes.items():
        year = int(n.year.iloc[0])
        b = runs[name]
        parts = [(k, g) for k, g in n.groupby("node")] + [("base", b[b.year == year])]
        for k, g in parts:
            rows.append(pd.DataFrame(dict(corpus=name, node=k, W=g.groupby("location").wake_share.first(), frames=g.groupby("location").n_frames.first(),
                                          gain=per_camera(g, "lazy_defer", "always_now", "value_score", SMALL))).reset_index())
    return pd.concat(rows, ignore_index=True)


def score_nodes(t):
    within = t.groupby(["corpus", "location"])[["W", "gain"]].apply(lambda g: spearmanr(g.W, g.gain)[0] if g.gain.nunique() > 1 else np.nan).dropna()
    m = t.groupby(["corpus", "node"]).agg(W=("W", "mean"), gain=("gain", "mean"))
    return dict(pairs=int(len(t)), rho=float(spearmanr(t.W, t.gain)[0]), rho_frames=float(spearmanr(t.frames, t.gain)[0]),
                within_median=float(within.median()), within_positive=float((within > 0).mean()), within_n=int(len(within)),
                within_p=greater(within.values), means={f"{a}/{b}": [float(x.W), float(x.gain)] for (a, b), x in m.iterrows()})


def verdicts(out):
    """The registered rules. n corpora; 'most' is n - 1 for H1 and H3 and 3 for H4 (2 if n < 5)."""
    names = [k for k in out if k not in ("pooled", "nodes")]
    n, P = len(names), out["pooled"]
    most, need4 = n - 1, 3 if n >= 5 else 2
    v = {}
    a = P["rho"] >= 0.5 and P["rho_p"] < 0.001
    b = sum(out[k]["rho"] > 0 for k in names) >= most
    c = P["n_hi"] >= 10 and P["n_lo"] >= 10 and P["gain_lazy_hi"] > 0 and P["gain_lazy_hi"] >= 3 * max(P["gain_lazy_lo"], 0.0)
    v["H1"] = dict(passed=bool(a and b and c), a_pooled=bool(a), b_within=bool(b), c_groups=bool(c),
                   c_testable=bool(P["n_hi"] >= 10 and P["n_lo"] >= 10))
    if "nodes" in out:
        N = out["nodes"]
        v["H1n"] = dict(passed=bool(N["rho"] >= 0.5 and N["rho"] - N["rho_frames"] >= 0.1 and N["within_median"] >= 0.5 and N["within_p"] < 0.001),
                        rho=N["rho"], rho_frames=N["rho_frames"], within_median=N["within_median"])
    v["H2"] = dict(passed=bool(P["gain_policy_hi"] >= 0.03 and P["gain_policy_hi_p"] < 0.001))
    p3 = holm([out[k]["life_p"] for k in names])
    ok3 = [bool(out[k]["life"] >= 1.5 and p < 0.05 and out[k]["dv_large"] >= -0.01) for k, p in zip(names, p3)]
    v["H3"] = dict(passed=sum(ok3) >= most, corpora=dict(zip(names, ok3)))
    if all("hv_all" in out[k] for k in names):
        p4 = holm([out[k]["hv_all_p"] for k in names])
        ok4 = [bool(out[k]["hv_all_rel"] >= 0.03 and p < 0.05) for k, p in zip(names, p4)]
        v["H4"] = dict(passed=sum(ok4) >= need4, corpora=dict(zip(names, ok4)), p_holm=dict(zip(names, p4)))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=path[+path2 ...]")
    ap.add_argument("--nodes", nargs="*", default=[], help="name=path of the qnode runs")
    ap.add_argument("--policy", default=POLICY)
    ap.add_argument("--front", nargs="+", default=FRONT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = {}
    for x in a.runs:
        name, path = x.split("=", 1)
        parts = [load(p) for p in path.split("+")]
        runs[name] = pd.concat([p[0] for p in parts], ignore_index=True)
        print(name, len(runs[name]), "runs,", sum(p[1] for p in parts), "errors")
    out, cams = score(runs, a.policy, a.front)
    if a.nodes:
        t = node_table(runs, {x.split("=", 1)[0]: load(x.split("=", 1)[1])[0] for x in a.nodes})
        t.to_csv(f"{a.out}/nodes.csv", index=False)
        out["nodes"] = score_nodes(t)
    v = verdicts(out)
    cams.to_csv(f"{a.out}/cameras.csv", index=False)
    with open(f"{a.out}/score_v3.json", "w") as f:
        json.dump(dict(scores=out, verdicts=v), f, indent=1)
    print(pd.DataFrame({k: x for k, x in out.items() if k != "nodes"}).round(4).to_string())
    if "nodes" in out:
        print(json.dumps(out["nodes"], indent=1))
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
