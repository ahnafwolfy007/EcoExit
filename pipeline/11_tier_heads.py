#!/usr/bin/env python
"""Gates D2, D3, D4 of PLAN_V2 on the cached tier features (development corpora).

    python pipeline/11_tier_heads.py --features DIR --tier-b b_mnv3l --tier-a a_mnv4s_kd --out outputs
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from sunsched.vision import crosstier as X
from sunsched.vision.context import HORIZON_S


def cam_value(pred, y, val, cams):
    return np.array([(val[cams == c] * (pred[cams == c] == y[cams == c])).sum() / val[cams == c].sum() for c in np.unique(cams)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--tier-a", default="a_mnv4s_kd")
    ap.add_argument("--tier-b", default="b_mnv3l")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = np.load(f"{a.features}/meta.npz")
    role, y, cams, t, order, val = (M[k] for k in ("role", "y", "cams", "t", "order", "val"))
    names = [str(x) for x in M["names"]]
    corpus = os.path.basename(a.features.rstrip("/")).replace("tier_features_", "")
    XA, XB = (np.load(f"{a.features}/X_{k}.npy") for k in (a.tier_a, a.tier_b))
    tr, ev = role == "train", role == "eval"
    ucam = np.unique(cams[ev])
    rows = []

    def rec(gate, variant, seed, avail, v):
        rows.extend(dict(corpus=corpus, gate=gate, variant=variant, seed=seed, available=avail, camera=c, value=x)
                    for c, x in zip(ucam, v))
        print(f"  {gate} {variant:<12} seed {seed} available {avail}: {v.mean():.4f}", flush=True)

    def data(sel, Xq, past, fut, stats=None):
        return X.Data(Xq[sel], XA[sel], XB[sel], y[sel], cams[sel], t[sel], order[sel], past, fut, dev, stats)

    nc = len(names)
    print("D2: tier B with memory (past 48 h and the next minute)")
    Dt = data(tr, XB, HORIZON_S, 60.0)
    De = data(ev, XB, HORIZON_S, 60.0, Dt.stats)
    g = torch.Generator().manual_seed(123)
    u = torch.rand(De.n_burst, generator=g).to(dev)[De.burst]
    for s in range(a.seeds):
        net = X.train(Dt, nc, None, seed=s)
        rec("D2", "plain", s, 1.0, cam_value(X.predict(net, De).argmax(1), y[ev], val[ev], cams[ev]))
        for mode in ("full", "mixed"):
            net = X.train(Dt, nc, mode, seed=s)
            for f in (1.0, 0.7, 0.4, 0.2, 0.0):
                rec("D2", mode, s, f, cam_value(X.predict(net, De, mode, u < f).argmax(1), y[ev], val[ev], cams[ev]))
            if mode == "mixed" and s == 0:
                torch.save(net.state_dict(), f"{a.out}/memory_head_{corpus}.pt")

    print("D3: tier A with causal memory (past 48 h only)")
    At = data(tr, XA, HORIZON_S, 0.0)
    Ae = data(ev, XA, HORIZON_S, 0.0, At.stats)
    for s in range(a.seeds):
        net = X.train(At, nc, None, seed=s)
        rec("D3", "plain", s, 1.0, cam_value(X.predict(net, Ae).argmax(1), y[ev], val[ev], cams[ev]))
        for mode in ("triage", "mixed"):
            net = X.train(At, nc, mode, seed=s)
            for f in ((1.0,) if mode == "triage" else (0.7, 0.4, 0.0)):
                rec("D3", mode, s, f, cam_value(X.predict(net, Ae, mode, u < f).argmax(1), y[ev], val[ev], cams[ev]))

    print("D4: which frames to refine (plain heads, no memory)")
    tc = np.unique(cams[tr])
    for s in range(a.seeds):
        fold = {c: i % 5 for i, c in enumerate(np.random.default_rng(s).permutation(tc))}
        f_of = np.array([fold[c] for c in cams[tr]])
        oof_a, oof_b = np.zeros((tr.sum(), nc), np.float32), np.zeros((tr.sum(), nc), np.float32)
        it = np.where(tr)[0]
        for k in range(min(5, len(tc))):
            fit, hold = it[f_of != k], it[f_of == k]
            for Xq, oof in ((XA, oof_a), (XB, oof_b)):
                d1 = data(fit, Xq, 0, 0)
                oof[f_of == k] = X.predict(X.train(d1, nc, None, seed=s), data(hold, Xq, 0, 0, d1.stats))
        za = X.predict(X.train(At, nc, None, seed=s), Ae)
        zb = X.predict(X.train(Dt, nc, None, seed=s), De)

        def dist(sel):
            x = XA[sel].astype(np.float32)
            x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
            out = np.zeros(len(x))
            for c in np.unique(cams[sel]):           # distance to the mean of the camera's EARLIER frames
                m = np.where(cams[sel] == c)[0]
                m = m[np.lexsort((order[sel][m], t[sel][m]))]
                past = np.cumsum(x[m], 0) - x[m]
                past /= np.linalg.norm(past, axis=1, keepdims=True) + 1e-9
                out[m] = 1 - (x[m] * past).sum(1)
            return out
        gain = val[tr] * ((oof_b.argmax(1) == y[tr]).astype(float) - (oof_a.argmax(1) == y[tr]).astype(float))
        gh = X.fit_gain(X.gain_features(oof_a, dist(tr)), gain, seed=s)(X.gain_features(za, dist(ev)))
        p = np.exp(za - za.max(1, keepdims=True))
        conf = (p / p.sum(1, keepdims=True)).max(1)
        va, vb = val[ev] * (za.argmax(1) == y[ev]), val[ev] * (zb.argmax(1) == y[ev])
        rng = np.random.default_rng(s)
        for f in (0.0, 0.3, 0.5, 1.0):
            for nm, r in (("gain_head", gh), ("confidence", -conf), ("random", rng.random(len(conf))), ("oracle", vb - va)):
                rec("D4", nm, s, f, X.value_at_budget(r, f, va, vb, val[ev], cams[ev]))
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/tier_heads_{corpus}.csv", index=False)


if __name__ == "__main__":
    main()
