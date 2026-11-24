"""Deferred-context classifier head (development, not part of any frozen test).

A frame is classified from its own trunk feature plus a small attention read
over features of other frames of the SAME camera. The read is a lightweight
relative of Context R-CNN (Beery et al., CVPR 2020), which keeps a per-camera
memory for a detector. What differs here is when the memory is available: a
node that classifies at capture time can only read the past ("causal"), while a
node that defers can also read frames captured after the one being classified.
The variants below separate that effect from the cheap alternative of waiting
one minute for the rest of a burst, which needs no solar scheduling at all.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunsched.vision.heads import class_weights

HORIZON_S = 48 * 3600.0
BURST_S = 60.0
K = 32

# name -> (past limit in s, future limit in s)
VARIANTS = {
    "none": (0.0, 0.0),
    "burst_causal": (BURST_S, 0.0),
    "burst_bi": (BURST_S, BURST_S),
    "causal": (HORIZON_S, 0.0),
    "causal_burst": (HORIZON_S, BURST_S),     # past memory plus a one-minute wait
    "bi": (HORIZON_S, HORIZON_S),
}


def build_index(cams, t, order, past, future, k=K, avail=None):
    """For every frame, up to k memory frames of the same camera.

    `order` breaks ties between frames with one timestamp (burst position).
    Half the slots go to the frames nearest in time, the rest are spread evenly
    over what remains of the window, so one long burst cannot fill the memory.
    Returns (idx, mask); idx is 0 where mask is False.
    """
    n = len(t)
    idx = np.zeros((n, k), dtype=np.int64)
    mask = np.zeros((n, k), dtype=bool)
    if past <= 0 and future <= 0:
        return idx, mask
    for c in np.unique(cams):
        m = np.where(cams == c)[0]
        m = m[np.lexsort((order[m], t[m]))]
        ts = t[m]
        ok = np.ones(len(m), dtype=bool) if avail is None else avail[m]
        for j in range(len(m)):
            lo = np.searchsorted(ts, ts[j] - past, side="left") if past > 0 else j
            hi = np.searchsorted(ts, ts[j] + future, side="right") if future > 0 else j + 1
            cand = np.r_[np.arange(lo, j), np.arange(j + 1, hi)]
            cand = cand[ok[cand]]
            if len(cand) > k:
                by_dt = cand[np.argsort(np.abs(ts[cand] - ts[j]), kind="stable")]
                near, rest = by_dt[:k // 2], np.sort(by_dt[k // 2:])
                pick = np.linspace(0, len(rest) - 1, k - k // 2).round().astype(int)
                cand = np.r_[near, rest[pick]]
            idx[m[j], :len(cand)] = m[cand]
            mask[m[j], :len(cand)] = True
    return idx, mask


def time_code(dt_s):
    """Signed time gap at several scales (Fourier features of the signed log gap,
    so the next frame and the one after it are told apart) plus the clock-hour
    difference between query and memory."""
    s = torch.sign(dt_s) * torch.log1p(dt_s.abs() / 10.0) / 10.0
    f = [s] + [g(np.pi * 2 ** k * s) for k in range(5) for g in (torch.sin, torch.cos)]
    ang = 2 * np.pi * dt_s / 86400.0
    return torch.stack(f + [torch.sin(ang), torch.cos(ang)], dim=-1)


TIME_DIM = 13


class ContextHead(nn.Module):
    def __init__(self, d_in, n_classes, d=128, hidden=256, dropout=0.2, heads=4, d_mem=None):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_classes))
        # Memory may hold another feature than the query's (d_mem), e.g. the triage
        # feature, which exists for every captured frame whether refined or not.
        self.q, self.kv = nn.Linear(d_in, d), nn.Linear(d_mem or d_in, d)
        self.t = nn.Sequential(nn.Linear(TIME_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.att = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.out = nn.Sequential(nn.Linear(3 * d, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_classes))
        nn.init.zeros_(self.out[-1].weight)       # starts exactly as the plain head
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, mem, dt, mask, own=None):
        logits = self.base(x)
        has = mask.any(1)
        if not bool(has.any()):
            return logits
        xq, mm, mk = self.q(x[has]), self.kv(mem[has]), mask[has]
        ctx, _ = self.att(xq.unsqueeze(1), mm + self.t(time_code(dt[has])), mm,
                          key_padding_mask=~mk, need_weights=False)
        w = mk.unsqueeze(-1).float()
        mean = (mm * w).sum(1) / w.sum(1)         # what this camera usually looks like
        z = self.out(torch.cat([xq, ctx.squeeze(1), self.kv((x if own is None else own)[has]) - mean], dim=1))
        full = torch.zeros_like(logits)
        full[has] = z
        return logits + full


def context_macs(d_in, n_classes, d=128, hidden=256, k=K):
    """MACs on top of the trunk and plain head (memory projections recomputed)."""
    return (k + 2) * d_in * d + 3 * d * k + 3 * k * d * d + 2 * d * d + 3 * d * hidden + hidden * n_classes


def _gather(Z, T, rows, idx, mask):
    mem = Z[idx[rows]]
    dt = (T[idx[rows]] - T[rows].unsqueeze(1)).float() * mask[rows]
    return mem, dt, mask[rows]


def train_context(X, y, cams, t, order, variant, n_classes, epochs=30, lr=1e-3, wd=1e-4,
                  seed=0, batch=256, device="cpu", mem_drop=0.3, M=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Z = torch.from_numpy(((X - mean) / std).astype(np.float32)).to(device)
    mstat = (mean, std) if M is None else (M.mean(0), M.std(0) + 1e-6)
    ZM = Z if M is None else torch.from_numpy(((M - mstat[0]) / mstat[1]).astype(np.float32)).to(device)
    T = torch.from_numpy(t.astype(np.float64)).to(device)
    Y = torch.from_numpy(y.astype(np.int64)).to(device)
    i_, m_ = build_index(cams, t, order, *VARIANTS[variant])
    idx, mask = torch.from_numpy(i_).to(device), torch.from_numpy(m_).to(device)
    head = ContextHead(X.shape[1], n_classes, d_mem=ZM.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = torch.from_numpy(class_weights(y, n_classes)).to(device)
    for _ in range(epochs):
        head.train()
        perm = rng.permutation(len(Z))
        for s in range(0, len(perm), batch):
            r = torch.from_numpy(perm[s:s + batch]).to(device)
            mem, dt, mk = _gather(ZM, T, r, idx, mask)
            mk = mk & (torch.rand(mk.shape, device=mk.device) > mem_drop)   # memory is never complete
            loss = F.cross_entropy(head(Z[r], mem, dt, mk, ZM[r]), Y[r], weight=w, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return head.eval(), mean, std, mstat


@torch.no_grad()
def predict(model, X, cams, t, order, past, future, device="cpu", batch=2048, M=None,
            avail=None, k=K):
    """`avail` marks the frames whose memory feature exists (default all)."""
    head, mean, std, mstat = model
    Z = torch.from_numpy(((X - mean) / std).astype(np.float32)).to(device)
    ZM = Z if M is None else torch.from_numpy(((M - mstat[0]) / mstat[1]).astype(np.float32)).to(device)
    T = torch.from_numpy(t.astype(np.float64)).to(device)
    i_, m_ = build_index(cams, t, order, past, future, k, avail)
    idx, mask = torch.from_numpy(i_).to(device), torch.from_numpy(m_).to(device)
    out = []
    for s in range(0, len(Z), batch):
        r = torch.arange(s, min(s + batch, len(Z)), device=device)
        mem, dt, mk = _gather(ZM, T, r, idx, mask)
        out.append(head(Z[r], mem, dt, mk, ZM[r]).cpu().numpy())
    return np.concatenate(out)
