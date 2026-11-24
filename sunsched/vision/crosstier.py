"""Mixed-fidelity per-camera memory over cached tier features (PLAN_V2, C2).

Every captured frame leaves a tier-A feature; a refined frame also leaves a
tier-B feature. A slot of the memory holds the tier-B feature when it exists
and the tier-A feature otherwise, each through its own projection, plus a
fidelity embedding and the time code of vision/context.py.

The plain head is trained first and frozen. The memory read adds a residual
that starts at zero, so a frame with an empty memory gets exactly the plain
head's answer. (Gate 2 found that a jointly trained head loses 0.017 on S10
when its memory is empty.)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunsched.vision.context import TIME_DIM, build_index, time_code
from sunsched.vision.heads import class_weights

MODES = ("full", "mixed", "triage")     # what a memory slot may hold


class MemoryHead(nn.Module):
    def __init__(self, d_q, d_a, d_b, nc, d=128, hidden=256, heads=4, dropout=0.2):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(d_q, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, nc))
        self.q, self.pa, self.pb = nn.Linear(d_q, d), nn.Linear(d_a, d), nn.Linear(d_b, d)
        self.fid = nn.Embedding(2, d)
        self.t = nn.Sequential(nn.Linear(TIME_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.out = nn.Sequential(nn.Linear(3 * d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, nc))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, ma, mb, hasb, dt, mask):
        logits = self.base(x)
        k = mask.any(1)
        if not bool(k.any()):
            return logits
        h = hasb[k].unsqueeze(-1)
        mem = self.norm(torch.where(h, self.pb(mb[k]), self.pa(ma[k])) + self.fid(hasb[k].long()))
        q = self.norm(self.q(x[k]))
        m = mask[k]
        ctx, _ = self.att(q.unsqueeze(1), mem + self.t(time_code(dt[k])), mem, key_padding_mask=~m, need_weights=False)
        w = m.unsqueeze(-1).float()
        mean = (mem * w).sum(1) / w.sum(1)
        res = torch.zeros_like(logits)
        res[k] = self.out(torch.cat([q, ctx.squeeze(1), q - mean], 1)).to(res.dtype)
        return logits + res


class Data:
    """Cached features of one role (rows of one split), on the device."""

    def __init__(self, Xq, Xa, Xb, y, cams, t, order, past, future, dev, stats=None):
        st = stats or [(X.mean(0).astype(np.float32), X.std(0).astype(np.float32) + 1e-6)
                       for X in (Xq.astype(np.float32), Xa.astype(np.float32), Xb.astype(np.float32))]
        self.stats, self.dev = st, dev
        z = lambda X, s: torch.from_numpy((X.astype(np.float32) - s[0]) / s[1]).to(dev)
        self.q, self.a, self.b = z(Xq, st[0]), z(Xa, st[1]), z(Xb, st[2])
        self.y = torch.from_numpy(y.astype(np.int64)).to(dev)
        self.t = torch.from_numpy(t.astype(np.float64)).to(dev)
        i, m = build_index(cams, t, order, past, future)
        self.idx, self.mask = torch.from_numpy(i).to(dev), torch.from_numpy(m).to(dev)
        burst = np.char.add(cams.astype(str), np.char.add("|", (t // 60).astype(np.int64).astype(str)))
        self.burst = torch.from_numpy(np.unique(burst, return_inverse=True)[1]).to(dev)
        self.n_burst = int(self.burst.max()) + 1 if len(y) else 0

    def batch(self, r, mode, avail):
        """avail: bool per row of this split, whether its tier-B feature exists."""
        idx = self.idx[r]
        mask, hasb = self.mask[r], avail[idx]
        if mode == "full":
            mask = mask & hasb
        if mode == "triage":
            hasb = torch.zeros_like(hasb)
        dt = (self.t[idx] - self.t[r].unsqueeze(1)).float() * mask
        return self.q[r], self.a[idx], self.b[idx], hasb, dt, mask

    def draw(self, p, gen):
        """Availability of whole bursts with probability p."""
        u = torch.rand(self.n_burst, generator=gen, device="cpu").to(self.dev)
        return (u < p)[self.burst]


def _fit(params, step, n, epochs, lr, seed, batch=256):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            loss = step(perm[s:s + batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()


def train(D: Data, nc, mode=None, epochs=30, mem_epochs=20, lr=1e-3, seed=0):
    """mode None: the plain head only. Otherwise the frozen plain head plus a memory read."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = MemoryHead(D.q.shape[1], D.a.shape[1], D.b.shape[1], nc).to(D.dev)
    w = torch.from_numpy(class_weights(D.y.cpu().numpy(), nc)).to(D.dev)
    ce = lambda z, r: F.cross_entropy(z, D.y[r], weight=w, label_smoothing=0.05)
    net.train()
    rows = lambda p: torch.from_numpy(p).to(D.dev)
    _fit(net.base.parameters(), lambda p: ce(net.base(D.q[rows(p)]), rows(p)), len(D.y), epochs, lr, seed)
    if mode is not None:
        for q in net.base.parameters():
            q.requires_grad_(False)
        net.base.eval()

        def step(p):
            r = rows(p)
            x, ma, mb, hb, dt, mk = D.batch(r, mode, D.draw(float(torch.rand(1, generator=gen)), gen))
            mk = mk & (torch.rand(mk.shape, device=mk.device) > 0.2)
            return ce(net(x, ma, mb, hb, dt, mk), r)
        _fit([q for n_, q in net.named_parameters() if not n_.startswith("base")], step, len(D.y), mem_epochs, lr, seed + 1)
    return net.eval()


@torch.no_grad()
def predict(net, D: Data, mode=None, avail=None, batch=2048):
    out = []
    av = torch.ones(len(D.y), dtype=torch.bool, device=D.dev) if avail is None else avail
    for s in range(0, len(D.y), batch):
        r = torch.arange(s, min(s + batch, len(D.y)), device=D.dev)
        if mode is None:
            out.append(net.base(D.q[r]).cpu().numpy())
        else:
            out.append(net(*D.batch(r, mode, av)).cpu().numpy())
    return np.concatenate(out)


# -- gain head (gate D4) ------------------------------------------------------------
def gain_features(za, xa_dist):
    """Per-frame inputs known after triage: tier-A probabilities summary and shift distance."""
    p = np.exp(za - za.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    s = np.sort(p, 1)
    ent = -(p * np.log(np.clip(p, 1e-9, 1))).sum(1)
    energy = -np.log(np.exp(za - za.max(1, keepdims=True)).sum(1)) - za.max(1)
    return np.concatenate([p, np.stack([s[:, -1], s[:, -1] - s[:, -2], ent, energy, xa_dist], 1)], 1).astype(np.float32)


def fit_gain(U, gain, seed=0, epochs=30):
    torch.manual_seed(seed)
    mu, sd = U.mean(0), U.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(U.shape[1], 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    X, g = torch.from_numpy((U - mu) / sd), torch.from_numpy(gain.astype(np.float32))
    net.train()
    _fit(net.parameters(), lambda p: F.mse_loss(net(X[p]).squeeze(1), g[p]), len(g), epochs, 1e-3, seed)
    net.eval()
    return lambda V: net(torch.from_numpy((V - mu) / sd)).squeeze(1).detach().numpy()


def value_at_budget(rank, frac, va, vb, w, cams):
    """Camera-mean value when the `frac` highest-ranked frames of each camera take tier B's answer."""
    out = []
    for c in np.unique(cams):
        m = np.where(cams == c)[0]
        k = int(round(frac * len(m)))
        take = m[np.argsort(-rank[m], kind="stable")[:k]]
        v = va[m].sum() + (vb[take] - va[take]).sum()
        out.append(v / w[m].sum())
    return np.array(out)
