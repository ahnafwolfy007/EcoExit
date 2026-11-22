"""Score a SunSched v3 candidate with the PREREGISTRATION_v2 rules.

    py scripts/score_v2.py --grid A_runs.csv [B_runs.csv ...] --wake W_runs.csv [...] \
        --policy sunsched3 --out data/derived/v3dev3 [--label cct20-dev]

The grid tables must together contain the candidate and the baselines in the
same cells (camera x year replicates). Rules (fixed in PREREGISTRATION_v2.md;
"Immediate" is always_now):

  HV1 near-best: in every cell with ratio >= 1, V = best mean value among the
      registered baselines, L = longest mean life among registered baselines
      with value >= V - 0.01. Near-best if value >= V - 0.01 and life >= 0.95 L.
      Holds if near-best in >= 75% of the cells. Descriptive.
  HV2 small storage: autonomy {0.5,1,2} x ratio {1,1.5,2}. Holds if the pooled
      mean value gain over Immediate is >= 0.02, the one-sided camera-level
      Wilcoxon gives Holm p < 0.05, and the mean gain is positive in >= 7 of 9
      cells.
  HV3 large storage: autonomy {30,100} x ratio {1.5,2,3}. Holds if the
      geometric-mean life ratio to Immediate is >= 1.15, the one-sided
      camera-level Wilcoxon on log life gives Holm p < 0.05, the ratio is
      >= 1.15 in >= 5 of 6 cells, and the pooled value change is >= -0.01.
  Holm: over the two pooled p-values of HV2 and HV3, within one corpus.
  HV4 no value cost: in every cell (all ratios), value >= Immediate - 0.01 in
      >= 90% of cells. Descriptive.
  HV5 cheap wakes: wake energies {1, 3} J, every wake cell; value >= Immediate
      - 0.01 in >= 75% of cells. Descriptive.
The camera is the unit: Wilcoxon on camera means over the family's cells and
the weather years, two-way (camera x year) bootstrap intervals. Per-cell tests
(HV2_per_cell_holm, HV3_per_cell_holm) are diagnostics, not decision rules.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEY = ["location", "year"]
CELL = ["autonomy", "ratio"]
BASELINES = ["always_now", "ee_now", "ceiling_now", "lazy_defer"]
ALL_RULES = BASELINES + ["defer_ceiling", "batch_anytime", "ceiling_conformal", "defer_ceiling_conformal"]
NB = 4000
rng = np.random.default_rng(0)


def load(paths):
    out = []
    for p in paths:
        d = pd.read_csv(p, low_memory=False)
        if "error" in d.columns:
            d = d[d.error.isna() | (d.error.astype(str) == "")]
        d["location"] = d["location"].astype(str)
        d = d.rename(columns={"tag_autonomy": "autonomy"})
        out.append(d)
    return pd.concat(out, ignore_index=True)


def holm(p):
    p = np.asarray(p, float)
    adj = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    q = p[ok][idx]
    m = len(q)
    run = 0.0
    res = np.empty(m)
    for i, v in enumerate(q):
        run = max(run, min(1.0, (m - i) * v))
        res[i] = run
    tmp = np.empty(m)
    tmp[idx] = res
    adj[ok] = tmp
    return adj


def paired(d, a, b, metric, how, cells):
    A = d[d.policy == a].set_index(cells + KEY)[metric]
    B = d[d.policy == b].set_index(cells + KEY)[metric]
    x = (np.log(A / B) if how == "logratio" else A - B).dropna().reset_index(name="d")
    rows = []
    for cell, g in x.groupby(cells):
        cell = cell if isinstance(cell, tuple) else (cell,)
        piv = g.pivot_table(index="location", columns="year", values="d").values
        li = rng.integers(0, piv.shape[0], (NB, piv.shape[0]))
        yi = rng.integers(0, piv.shape[1], (NB, piv.shape[1]))
        boots = np.array([np.nanmean(piv[np.ix_(l, y)]) for l, y in zip(li, yi)])
        lo, hi = np.quantile(boots, [0.025, 0.975])
        cam = g.groupby("location").d.mean()
        p = float(wilcoxon(cam).pvalue) if len(cam) >= 6 and not np.allclose(cam, 0) else np.nan
        m = float(g.d.mean())
        if how == "logratio":
            m, lo, hi = np.exp(m), np.exp(lo), np.exp(hi)
        rows.append(dict(zip(cells, cell), n=len(g), n_cameras=len(cam), cameras_positive=int((cam > 0).sum()),
                         effect=m, ci_lo=float(lo), ci_hi=float(hi), p=p))
    return pd.DataFrame(rows)


def pooled(d, a, b, metric, how, cells):
    """One confirmatory test per family: each camera's mean paired effect over
    all the family's cells and years, then a one-sided exact Wilcoxon (> 0)."""
    A = d[d.policy == a].set_index(cells + KEY)[metric]
    B = d[d.policy == b].set_index(cells + KEY)[metric]
    x = (np.log(A / B) if how == "logratio" else A - B).dropna().reset_index(name="d")
    cam = x.groupby("location").d.mean()
    p = float(wilcoxon(cam, alternative="greater").pvalue) if not np.allclose(cam, 0) else 1.0
    m = float(x.d.mean())
    piv = x.groupby(["location", "year"]).d.mean().unstack().values
    li = rng.integers(0, piv.shape[0], (NB, piv.shape[0]))
    yi = rng.integers(0, piv.shape[1], (NB, piv.shape[1]))
    boots = np.array([np.nanmean(piv[np.ix_(l, y)]) for l, y in zip(li, yi)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    if how == "logratio":
        m, lo, hi = np.exp(m), np.exp(lo), np.exp(hi)
    return dict(effect=m, ci_lo=float(lo), ci_hi=float(hi), n_cameras=int(len(cam)),
                cameras_positive=int((cam > 0).sum()), p_one_sided=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs="+", required=True)
    ap.add_argument("--wake", nargs="*", default=[])
    ap.add_argument("--policy", default="sunsched3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--hv3-ratios", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = a.policy
    g = load(a.grid)
    S = dict(label=a.label, policy=P, rows=int(len(g)))

    means = g.groupby(CELL + ["policy"])[["value_score", "battery_years_to_eol", "ceiling_mean",
                                         "frac_time_soc_above_90", "triage_final_frac", "refined_now_frac",
                                         "refined_later_frac", "dropped_frac", "b_wakes_per_day",
                                         "latency_p95_h"]].mean().reset_index()
    means.to_csv(os.path.join(a.out, f"means_{P}.csv"), index=False)

    # -- HV1 near-best --------------------------------------------------------------
    rows = []
    for (au, r), sub in means[means.ratio >= 1].groupby(CELL):
        s = sub.set_index("policy")
        if P not in s.index:
            continue
        for ref_name, ref in (("registered", BASELINES), ("all_rules", ALL_RULES)):
            ref = [x for x in ref if x in s.index]
            V = s.loc[ref].value_score.max()
            ok = s.loc[ref][s.loc[ref].value_score >= V - 0.01]
            L = ok.battery_years_to_eol.max()
            v, l = s.loc[P].value_score, s.loc[P].battery_years_to_eol
            rows.append(dict(autonomy=au, ratio=r, reference=ref_name, best=s.loc[ref].value_score.idxmax(),
                             V=V, L=L, value=v, life=l, dv=v - V, life_frac=l / L,
                             near_best=bool(v >= V - 0.01 and l >= 0.95 * L)))
    hv1 = pd.DataFrame(rows)
    hv1.to_csv(os.path.join(a.out, f"hv1_{P}.csv"), index=False)
    for ref_name, t in hv1.groupby("reference"):
        S[f"HV1_{ref_name}"] = dict(cells=int(len(t)), near_best=int(t.near_best.sum()),
                                    share=float(t.near_best.mean()), holds=bool(t.near_best.mean() >= 0.75),
                                    misses=t[~t.near_best][["autonomy", "ratio", "best", "dv", "life_frac"]]
                                    .round(4).to_dict("records"))

    # Same rule applied to every policy (a fixed rule is compared with the
    # registered baselines including itself): which single rule adapts best?
    share = {}
    for pol in means.policy.unique():
        hits = []
        for (au, r), sub in means[means.ratio >= 1].groupby(CELL):
            s = sub.set_index("policy")
            ref = [x for x in BASELINES if x in s.index]
            V = s.loc[ref].value_score.max()
            L = s.loc[ref][s.loc[ref].value_score >= V - 0.01].battery_years_to_eol.max()
            if pol in s.index:
                hits.append(s.loc[pol].value_score >= V - 0.01 and s.loc[pol].battery_years_to_eol >= 0.95 * L)
        share[pol] = round(float(np.mean(hits)), 3)
    S["near_best_share_by_policy"] = dict(sorted(share.items(), key=lambda kv: -kv[1]))

    # -- HV2 small storage ----------------------------------------------------------
    sm = g[g.autonomy.isin([0.5, 1, 2]) & g.ratio.isin([1, 1.5, 2])]
    hv2 = paired(sm, P, "always_now", "value_score", "diff", CELL)
    hv2["p_holm"] = holm(hv2.p)
    hv2["pass"] = (hv2.effect >= 0.02) & (hv2.p_holm < 0.05)
    hv2.to_csv(os.path.join(a.out, f"hv2_{P}.csv"), index=False)
    need2 = 5 if len(hv2) == 9 else int(np.ceil(5 / 9 * len(hv2)))
    S["HV2_per_cell_holm"] = dict(cells=int(len(hv2)), passed=int(hv2["pass"].sum()), needed=need2,
                                  holds=bool(hv2["pass"].sum() >= need2),
                                  table=hv2[CELL + ["effect", "ci_lo", "ci_hi", "cameras_positive", "p_holm",
                                                    "pass"]].round(4).to_dict("records"))
    pz = pooled(sm, P, "always_now", "value_score", "diff", CELL)
    pos = int((hv2.effect > 0).sum())
    need_pos2 = int(np.ceil(7 / 9 * len(hv2)))
    S["HV2"] = dict(**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in pz.items()},
                    cells=int(len(hv2)), cells_positive=pos, cells_positive_needed=need_pos2,
                    criteria_except_p=bool(pz["effect"] >= 0.02 and pos >= need_pos2))

    # -- HV3 large storage ----------------------------------------------------------
    lg = g[g.autonomy.isin([30, 100]) & g.ratio.isin([1, 1.5, 2, 3])]
    life = paired(lg, P, "always_now", "battery_years_to_eol", "logratio", CELL)
    val = paired(lg, P, "always_now", "value_score", "diff", CELL)[CELL + ["effect"]].rename(
        columns={"effect": "dvalue"})
    hv3 = life.merge(val, on=CELL)
    hv3["p_holm"] = holm(hv3.p)
    hv3["pass"] = (hv3.effect >= 1.15) & (hv3.p_holm < 0.05) & (hv3.dvalue >= -0.01)
    hv3.to_csv(os.path.join(a.out, f"hv3_{P}.csv"), index=False)
    need3 = 4 if len(hv3) == 6 else int(np.ceil(4 / 6 * len(hv3)))
    S["HV3_per_cell_holm"] = dict(cells=int(len(hv3)), passed=int(hv3["pass"].sum()), needed=need3,
                                  holds=bool(hv3["pass"].sum() >= need3),
                                  table=hv3[CELL + ["effect", "ci_lo", "ci_hi", "dvalue", "p_holm", "pass"]]
                                  .round(4).to_dict("records"))
    for name, rset in (("HV3", a.hv3_ratios), ("HV3_ratio1", [1.0])):
        sub = lg[lg.ratio.isin(rset)]
        if sub.empty:
            continue
        pl = pooled(sub, P, "always_now", "battery_years_to_eol", "logratio", CELL)
        pv = pooled(sub, P, "always_now", "value_score", "diff", CELL)
        h = hv3[hv3.ratio.isin(rset)]
        S[name] = dict(ratios=rset, life_ratio=round(pl["effect"], 4), life_ci=[round(pl["ci_lo"], 4),
                       round(pl["ci_hi"], 4)], cameras_longer=pl["cameras_positive"], n_cameras=pl["n_cameras"],
                       p_one_sided=pl["p_one_sided"], dvalue=round(pv["effect"], 4),
                       dvalue_ci=[round(pv["ci_lo"], 4), round(pv["ci_hi"], 4)],
                       cells_life_ge_1p15=int((h.effect >= 1.15).sum()), cells=int(len(h)),
                       cells_needed=int(np.ceil(5 / 6 * len(h))),
                       criteria_except_p=bool(pl["effect"] >= 1.15 and pv["effect"] >= -0.01
                                              and (h.effect >= 1.15).sum() >= np.ceil(5 / 6 * len(h))))

    # Holm over the two confirmatory tests of this corpus (HV2, HV3).
    fam = [k for k in ("HV2", "HV3") if k in S]
    adj = holm([S[k]["p_one_sided"] for k in fam])
    for k, q in zip(fam, adj):
        S[k]["p_holm"] = float(q)
        S[k]["holds"] = bool(S[k]["criteria_except_p"] and q < 0.05)
    if "HV3_ratio1" in S:                    # exploratory, not in the Holm family
        S["HV3_ratio1"]["holds_uncorrected"] = bool(S["HV3_ratio1"]["criteria_except_p"]
                                                    and S["HV3_ratio1"]["p_one_sided"] < 0.05)

    # -- HV4 no value cost against Immediate, every cell -----------------------------
    pv_ = means.pivot_table(index=CELL, columns="policy", values="value_score")
    if P in pv_.columns and "always_now" in pv_.columns:
        d4 = (pv_[P] - pv_["always_now"]).dropna().rename("d_now").reset_index()
        d4.to_csv(os.path.join(a.out, f"hv4_{P}.csv"), index=False)
        S["HV4"] = dict(cells=int(len(d4)), passed=int((d4.d_now >= -0.01).sum()),
                        share=float((d4.d_now >= -0.01).mean()), holds=bool((d4.d_now >= -0.01).mean() >= 0.90),
                        worst=d4.sort_values("d_now").head(4).round(4).to_dict("records"))

    # -- HV5 cheap wakes ------------------------------------------------------------
    if a.wake:
        w = load(a.wake)
        w = w[w.tag_param == "node.tier_b_wake_j"].rename(columns={"tag_value": "wake_j"})
        wm = w.groupby(["wake_j"] + CELL + ["policy"]).value_score.mean().unstack("policy").reset_index()
        if P in wm.columns:
            wm["d_now"] = wm[P] - wm["always_now"]
            if "lazy_defer" in wm.columns:
                wm["d_defer"] = wm[P] - wm["lazy_defer"]
            wm.to_csv(os.path.join(a.out, f"hv5_{P}.csv"), index=False)
            c = wm[wm.wake_j.isin([1, 3])]
            S["HV5"] = dict(cells=int(len(c)), passed=int((c.d_now >= -0.01).sum()),
                            share=float((c.d_now >= -0.01).mean()), holds=bool((c.d_now >= -0.01).mean() >= 0.75),
                            worst=c.sort_values("d_now").head(4)[["wake_j"] + CELL + ["d_now"]].round(4)
                            .to_dict("records"))
            S["wake_all_vs_now"] = dict(cells=int(len(wm)), within_0p01=int((wm.d_now >= -0.01).sum()),
                                        min=float(wm.d_now.min()), mean=float(wm.d_now.mean()))
            if "d_defer" in wm:
                S["wake_all_vs_defer"] = dict(within_0p01=int((wm.d_defer >= -0.01).sum()),
                                              min=float(wm.d_defer.min()), mean=float(wm.d_defer.mean()))

    # -- overview -------------------------------------------------------------------
    ov = means.groupby("policy").value_score.mean().sort_values(ascending=False)
    S["mean_value_all_cells"] = ov.round(4).to_dict()
    with open(os.path.join(a.out, f"summary_{P}.json"), "w") as f:
        json.dump(S, f, indent=1, default=float)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "table"})
                      for k, v in S.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
