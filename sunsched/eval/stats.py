"""Intervals, paired tests and multiple-comparison correction.

Replicates are (camera location x weather year) pairs, and every policy sees
the identical replicate, so comparisons are paired. With 9 locations and 5
years there are 45 pairs per condition. That matters: a Wilcoxon test on 5
pairs can never reach p < 0.05 (its smallest possible two-sided p is 0.0625),
which the previous version of this project's protocol overlooked.
"""
from typing import Dict, List, Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], level: float = 0.95, n_boot: int = 2000,
                 seed: int = 0):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    out = dict(n=int(len(d)), mean_diff=float(d.mean()) if len(d) else float("nan"),
               median_diff=float(np.median(d)) if len(d) else float("nan"))
    if len(d) < 6 or np.allclose(d, 0.0):
        out["p_value"] = 1.0 if len(d) and np.allclose(d, 0.0) else float("nan")
        return out
    from scipy.stats import wilcoxon
    out["p_value"] = float(wilcoxon(a, b).pvalue)
    return out


def holm(p_values: Sequence[float], alpha: float = 0.05) -> List[Dict[str, float]]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    m = int(np.isfinite(p).sum())
    out = [dict(p_adjusted=float("nan"), reject=False) for _ in p]
    running = 0.0
    for rank, i in enumerate(order):
        if not np.isfinite(p[i]):
            continue
        adj = min(1.0, max(running, (m - rank) * p[i]))
        running = adj
        out[i] = dict(p_adjusted=float(adj), reject=bool(adj < alpha))
    return out


def aggregate(rows: List[dict], keys: Sequence[str], metrics: Sequence[str],
              level: float = 0.95, n_boot: int = 2000) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = dict(zip(keys, key))
        row["n"] = len(rs)
        for m in metrics:
            mean, lo, hi = bootstrap_ci([r.get(m, np.nan) for r in rs], level, n_boot)
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
        out.append(row)
    return out
