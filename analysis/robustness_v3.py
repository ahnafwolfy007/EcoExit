"""Robustness of H1 and H1n on the frozen test (exploratory, not registered).

    py robustness.py CAMERAS_CSV DEV_CAMERAS_CSV NODES_CSV OUT_JSON

Nothing here changes a registered verdict. CIs resample cameras inside each corpus.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

B, rng = 4000, np.random.default_rng(0)
sp = lambda x, y: float(spearmanr(x, y)[0])
q = lambda v: [float(x) for x in np.nanquantile(v, [.025, .975])]


def strata(d, key="corpus"):
    g = [x.index.values for _, x in d.groupby(key)]
    return lambda: np.concatenate([rng.choice(i, len(i)) for i in g])


def ranked(d, col):
    return d.groupby("corpus")[col].rank()


def auc(s, l):
    r, n1, n0 = rankdata(s), l.sum(), (~l).sum()
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    c, dev, nodes = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2]), pd.read_csv(sys.argv[3])
    out = {}
    W, F, G = c.W.values, c.frames.values, c.gain_lazy.values
    draw = strata(c)
    boot = np.array([(sp(W[i], G[i]), sp(F[i], G[i])) for i in (draw() for _ in range(B))])
    out["pooled"] = dict(W=sp(W, G), W_ci=q(boot[:, 0]), frames=sp(F, G), frames_ci=q(boot[:, 1]), diff_ci=q(boot[:, 0] - boot[:, 1]))
    out["leave_one_out"] = {k: dict(W=sp(g.W, g.gain_lazy), frames=sp(g.frames, g.gain_lazy)) for k, g in ((k, c[c.corpus != k]) for k in c.corpus.unique())}
    out["within"] = {}
    for k, g in c.groupby("corpus"):
        b = [sp(g.W.values[j], g.gain_lazy.values[j]) for j in (rng.choice(len(g), len(g)) for _ in range(B))]
        out["within"][k] = dict(W=sp(g.W, g.gain_lazy), W_ci=q(b), frames=sp(g.frames, g.gain_lazy))
    # ranks inside each corpus, then pooled: what is left when corpus differences are taken out
    r = c.assign(rw=ranked(c, "W"), rg=ranked(c, "gain_lazy"), rf=ranked(c, "frames"))
    cc = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    b2 = []
    for _ in range(B):
        x = c.iloc[draw()]
        a, g2, f2 = (x.groupby("corpus")[k].rank() for k in ("W", "gain_lazy", "frames"))
        b2.append((cc(a, g2), cc(f2, g2)))
    b2 = np.array(b2)
    perm = [cc(r.rw, r.groupby("corpus").rg.transform(lambda x: rng.permutation(x.values))) for _ in range(5000)]
    obs = cc(r.rw, r.rg)
    out["corpus_adjusted"] = dict(W=obs, W_ci=q(b2[:, 0]), frames=cc(r.rf, r.rg), frames_ci=q(b2[:, 1]),
                                  perm_p=float((1 + (np.array(perm) >= obs).sum()) / 5001))
    # nodes: resample cameras, keep all of a camera's nodes together
    n = nodes[nodes.node != "deadline12"].reset_index(drop=True)
    n["cam"] = n.corpus + "|" + n.location.astype(str)
    where = {k: v.index.values for k, v in n.groupby("cam")}
    per = [g.cam.unique() for _, g in n.groupby("corpus")]
    nb = []
    for _ in range(2000):
        i = np.concatenate([where[k] for k in np.concatenate([rng.choice(p, len(p)) for p in per])])
        nb.append((sp(n.W.values[i], n.gain.values[i]), sp(n.frames.values[i], n.gain.values[i])))
    nb = np.array(nb)
    wi = n.groupby("cam")[["W", "gain"]].apply(lambda g: sp(g.W, g.gain) if g.gain.nunique() > 1 else np.nan).dropna()
    out["nodes"] = dict(W=sp(n.W, n.gain), W_ci=q(nb[:, 0]), frames=sp(n.frames, n.gain), frames_ci=q(nb[:, 1]), diff_ci=q(nb[:, 0] - nb[:, 1]),
                        loo_W=[sp(g.W, g.gain) for g in (n[n.corpus != k] for k in n.corpus.unique())],
                        within_mean=float(wi.mean()), within_mean_ci=q([rng.choice(wi.values, len(wi)).mean() for _ in range(B)]))
    # ranking is not prediction: a line fitted on the development cameras
    A = np.c_[np.ones(len(dev)), dev.W]
    beta = np.linalg.lstsq(A, dev.gain_lazy.values, rcond=None)[0]
    p = np.c_[np.ones(len(c)), W] @ beta
    out["transfer"] = dict(bias=float((p - G).mean()), R2_test_mean=float(1 - ((p - G) ** 2).sum() / ((G.mean() - G) ** 2).sum()),
                           mae=float(np.abs(p - G).mean()))
    l = G >= 0.02
    out["select"] = dict(n_pos=int(l.sum()), auc_W=auc(W, l), auc_frames=auc(F, l),
                         **{f"rule_{t}": dict(precision=float((l & (W >= t)).sum() / (W >= t).sum()), recall=float((l & (W >= t)).sum() / l.sum())) for t in (0.02, 0.05)})
    json.dump(out, open(sys.argv[4], "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
