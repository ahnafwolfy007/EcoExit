"""Statistical protocol: seeds, intervals, paired tests, multiplicity.

The original pipeline reported single-seed point estimates with no intervals
and no tests, across eight policies and five sites. With that many arms, a
favourable-looking gap is the default outcome of noise. Everything here exists
so a headline number comes with an interval and a test, and so the ablation
matrix is corrected for the number of comparisons it makes.
"""
from typing import Dict, List, Sequence, Tuple
import numpy as np


def bootstrap_ci(values: Sequence[float], level: float = 0.95,
                 n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """Percentile bootstrap. Returns (mean, lo, hi).

    Used rather than a t interval because seed-to-seed distributions here are
    small-n and visibly skewed (a single brownout day moves a run a long way).
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean()
                      for _ in range(n_boot)])
    a = (1.0 - level) / 2.0
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def aggregate_seeds(rows: List[dict], key_fields: Sequence[str],
                    metric_fields: Sequence[str], level: float = 0.95,
                    n_boot: int = 2000) -> List[dict]:
    """Collapse per-seed rows into one row per condition, with intervals."""
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in key_fields), []).append(r)

    out = []
    for key, rs in groups.items():
        agg = dict(zip(key_fields, key))
        agg["n_seeds"] = len(rs)
        for m in metric_fields:
            vals = [r[m] for r in rs if m in r and r[m] is not None]
            mean, lo, hi = bootstrap_ci(vals, level=level, n_boot=n_boot)
            agg[m] = mean
            agg[f"{m}_lo"] = lo
            agg[f"{m}_hi"] = hi
            agg[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(agg)
    return out


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test of a vs b.

    Paired because every policy sees the identical trace, stream and seed, so
    the seed effect is common and should be differenced out. This is the test
    that has to be run against `static_max`, since that is the baseline
    currently winning.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return dict(statistic=float("nan"), p_value=float("nan"), n=n,
                    median_diff=float("nan"))
    diff = a[:n] - b[:n]
    try:
        from scipy.stats import wilcoxon
        if np.allclose(diff, 0.0):
            return dict(statistic=0.0, p_value=1.0, n=n, median_diff=0.0)
        stat, p = wilcoxon(a[:n], b[:n])
        return dict(statistic=float(stat), p_value=float(p), n=n,
                    median_diff=float(np.median(diff)))
    except ImportError:
        # Sign test fallback, so the protocol still runs without scipy.
        pos = int((diff > 0).sum())
        neg = int((diff < 0).sum())
        k = min(pos, neg)
        tot = pos + neg
        if tot == 0:
            return dict(statistic=0.0, p_value=1.0, n=n, median_diff=0.0)
        from math import comb
        p = min(1.0, 2.0 * sum(comb(tot, i) for i in range(k + 1)) / (2 ** tot))
        return dict(statistic=float(k), p_value=float(p), n=n,
                    median_diff=float(np.median(diff)))


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> List[dict]:
    """Holm step-down correction across the ablation arms.

    Preferred over plain Bonferroni: uniformly more powerful at the same
    family-wise error rate, and it costs one sort.
    """
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    out = [None] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, max(prev, (m - rank) * p[idx]))
        prev = adj
        out[idx] = dict(p_raw=float(p[idx]), p_adjusted=float(adj),
                        reject=bool(adj < alpha))
    return out


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> float:
    """Class-balanced F1.

    With 51.7% of camera-trap frames empty, raw accuracy is close to
    meaningless -- a model that predicts "empty" always scores above half.
    Macro-F1 and per-class recall are the defensible numbers.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fn == 0:
            continue               # class absent from this sample
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else float("nan")


def compare_against_baseline(per_seed: Dict[str, List[float]], baseline: str,
                             alpha: float = 0.05) -> List[dict]:
    """Every policy against one baseline, Holm-corrected in a single family."""
    if baseline not in per_seed:
        raise KeyError(f"baseline {baseline!r} not among {sorted(per_seed)}")
    others = [k for k in per_seed if k != baseline]
    tests = [paired_wilcoxon(per_seed[k], per_seed[baseline]) for k in others]
    corrected = holm_bonferroni([t["p_value"] for t in tests], alpha=alpha)
    return [dict(policy=k, baseline=baseline, **t, **c)
            for k, t, c in zip(others, tests, corrected)]
