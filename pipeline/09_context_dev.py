#!/usr/bin/env python
"""Development gate for a stronger classifier. Vision only, no simulation.

    python pipeline/09_context_dev.py --corpus cct20 --device cuda

Two questions, on corpora that have already been seen:
  1. Does fine-tuning the trunk beat the frozen trunk on unseen cameras?
  2. Does memory from AFTER the capture help beyond (a) the past and
     (b) a one-minute wait for the rest of the burst?
The adoption rule is in Paper/sunsched-paper/REVISION_RESULTS.md section 15 and
was written before this script was run.
"""
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sunsched.cli import banner, base_parser, load_cfg, write_csv
from sunsched.data import cct20, source
from sunsched.vision import context as C
from sunsched.vision import device as vision_device
from sunsched.vision.backbone import IMAGENET_MEAN, IMAGENET_STD, TapExtractor, input_size, load_trunk, pool
from sunsched.vision.heads import class_weights

WAITS = [("past_only", 0.0), ("1min", 60.0), ("1h", 3600.0), ("6h", 6 * 3600.0),
         ("24h", 24 * 3600.0), ("48h", 48 * 3600.0)]


class Jpegs(Dataset):
    def __init__(self, blobs, rows, train):
        self.b, self.r, self.train = blobs, rows, train

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        a = np.array(Image.open(io.BytesIO(self.b[self.r[i]])).convert("RGB"))
        x = torch.from_numpy(a).permute(2, 0, 1)
        if self.train:
            if torch.rand(1) < 0.5:
                x = x.flip(2)
            if torch.rand(1) < 0.2:                      # infra-red night frames are grey
                x = x.float().mean(0, keepdim=True).expand(3, -1, -1).to(torch.uint8)
        return x, self.r[i]


def norm(x, dev):
    x = x.to(dev, non_blocking=True).float() / 255.0
    return (x - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)


def finetune(trunk, taps, blobs, rows, y, nc, small, dev, epochs, seed=0):
    """Joint multi-exit fine-tuning: triage (low res, first tap), lite and full
    (high res, later taps). BatchNorm statistics stay at their ImageNet values."""
    torch.manual_seed(seed)
    for p in trunk.parameters():
        p.requires_grad_(True)
    net = nn.ModuleList(list(trunk))
    with torch.no_grad():
        x, dims = torch.zeros(1, 3, *small, device=dev), {}
        for i, l in enumerate(trunk):
            x = l(x)
            if i in taps:
                dims[i] = x.shape[1]
    ch = {t: nn.Linear(2 * dims[t], nc).to(dev) for t in taps}
    ch_lo = nn.Linear(2 * dims[taps[0]], nc).to(dev)
    heads = list(ch.values()) + [ch_lo]
    opt = torch.optim.AdamW([dict(params=net.parameters(), lr=1e-4),
                             dict(params=[p for h in heads for p in h.parameters()], lr=1e-3)],
                            weight_decay=1e-2)
    dl = DataLoader(Jpegs(blobs, rows, True), batch_size=64, shuffle=True, num_workers=4,
                    drop_last=True, persistent_workers=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [1e-4, 1e-3], total_steps=epochs * len(dl))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    w = torch.from_numpy(class_weights(y[rows], nc)).to(dev)
    Y = torch.from_numpy(y).to(dev)

    def run(x, upto):
        o = {}
        for i, l in enumerate(trunk):
            x = l(x)
            if i in taps:
                o[i] = pool(x, grid=False)
            if i == upto:
                break
        return o

    for ep in range(epochs):
        trunk.train()
        for m in trunk.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        tot, seen, t0 = 0.0, 0, time.time()
        for xb, r in dl:
            xb, yb = norm(xb, dev), Y[r.to(dev)]
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                o = run(xb, taps[-1])
                lo = run(F.interpolate(xb, size=small, mode="area"), taps[0])
                ce = lambda z: F.cross_entropy(z, yb, weight=w, label_smoothing=0.05)
                loss = ce(ch[taps[2]](o[taps[2]])) + 0.5 * ce(ch[taps[1]](o[taps[1]])) \
                    + 0.5 * ce(ch_lo(lo[taps[0]]))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, seen = tot + float(loss) * len(r), seen + len(r)
        print(f"    epoch {ep + 1}/{epochs}  loss {tot / seen:.3f}  {time.time() - t0:.0f} s", flush=True)
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk.eval()


def extract(trunk, taps, blobs, rows, small, dev):
    ext = TapExtractor(trunk, taps).eval().to(dev)
    dl = DataLoader(Jpegs(blobs, rows, False), batch_size=64, num_workers=4)
    full, tri = [], []
    with torch.no_grad():
        for xb, _ in dl:
            xb = norm(xb, dev)
            full.append(ext(xb)[-1].cpu().numpy().astype(np.float32))
            tri.append(ext(F.interpolate(xb, size=small, mode="area"))[0].cpu().numpy().astype(np.float32))
    return np.concatenate(full), np.concatenate(tri)


def per_camera(pred, y, val, cams):
    rows = []
    for c in np.unique(cams):
        m = cams == c
        rows.append((c, int(m.sum()), float((pred[m] == y[m]).mean()),
                     float((val[m] * (pred[m] == y[m])).sum() / val[m].sum())))
    return rows


def gate2(args, cfg, names, full, tri, tr, ev, y, cams, t, order, val, dev):
    """Past memory plus a one-minute burst wait, with realistic memory contents.

    A deferring node holds full features only of the frames it refined, so
    memory made of full features is re-tested with 70, 40 and 20 % of frames
    available (random whole bursts, fixed seed). Memory made of triage features
    exists for every captured frame and is the alternative."""
    rows = []
    past, fut = C.VARIANTS["causal_burst"]
    burst = np.char.add(cams.astype(str), (t // C.BURST_S).astype(np.int64).astype(str))
    _, b = np.unique(burst, return_inverse=True)
    u = np.random.default_rng(0).random(b.max() + 1)[b]
    for seed in range(args.seeds):
        for mem, M, variant in (("none", None, "none"), ("full", None, "causal_burst"),
                                ("triage", tri, "causal_burst")):
            model = C.train_context(full[tr], y[tr], cams[tr], t[tr], order[tr], variant, len(names), seed=seed,
                                    device=dev, M=None if M is None else M[tr])
            for frac in ([1.0, 0.7, 0.4, 0.2, 0.0] if mem == "full" else [1.0]):
                p_, f_ = C.VARIANTS["none"] if mem == "none" else (past, fut)
                pred = C.predict(model, full[ev], cams[ev], t[ev], order[ev], p_, f_, device=dev,
                                 M=None if M is None else M[ev], avail=(u < frac)[ev]).argmax(1)
                pc = per_camera(pred, y[ev], val[ev], cams[ev])
                rows += [dict(corpus=cfg.data.corpus, memory=mem, available=frac, seed=seed, camera=c, n=n,
                              acc=a, value=vs) for c, n, a, vs in pc]
                print(f"  seed {seed} memory {mem:<6} available {frac:.1f}: camera value "
                      f"{np.mean([x[3] for x in pc]):.3f}", flush=True)
        write_csv(f"{cfg.results_dir}/tables/context_gate2.csv", rows)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gate2", action="store_true", help="fine-tuned trunk only; what may the memory hold?")
    ap.add_argument("--ft-epochs", type=int, default=8)
    args = ap.parse_args()
    cfg = load_cfg(args)
    v, dev = cfg.vision, vision_device.resolve(cfg.vision.device)
    h, w = input_size(v.resolutions[1], v.aspect)
    small = input_size(v.resolutions[0], v.aspect)

    banner(f"1. images ({cfg.data.corpus})")
    splits, names = source.load_splits(cfg, quick=args.quick)
    recs = [(s, r) for s in source.used_splits(cfg) for r in splits[s]]
    where = {r.file_name: i for i, (_, r) in enumerate(recs)}
    blobs = [None] * len(recs)
    for name, jpeg in source.image_source(cfg, where.keys(), ""):
        try:
            b = io.BytesIO()
            cct20.decode(jpeg, h, w).resize((w, h)).save(b, "JPEG", quality=92)
            blobs[where[name]] = b.getvalue()
        except Exception:
            continue
    keep = [i for i, b in enumerate(blobs) if b is not None]
    recs, blobs = [recs[i] for i in keep], [blobs[i] for i in keep]
    role = np.array(["train" if s in cfg.data.train_splits else
                     "calib" if s == cfg.data.calib_split else "eval" for s, _ in recs])
    y = np.array([names.index(r.label) for _, r in recs], dtype=np.int64)
    cams = np.array([str(r.location) for _, r in recs])
    t = np.array([r.timestamp.timestamp() for _, r in recs])
    order = np.array([r.frame_num for _, r in recs])
    val = np.array([cfg.task.value_empty if names[k] == cfg.task.empty_class else
                    cfg.task.value_vehicle if names[k] in cfg.task.vehicle_classes else
                    cfg.task.value_animal for k in y])
    idx = {k: np.where(role == k)[0] for k in ("train", "calib", "eval")}
    print({k: (len(i), len(set(cams[i]))) for k, i in idx.items()}, f"{len(names)} classes")

    rows, summary = [], []
    tr, ev = idx["train"], idx["eval"]
    allrows = np.arange(len(recs))
    for trunk_kind in (("finetuned",) if args.gate2 else ("frozen", "finetuned")):
        banner(f"2. trunk: {trunk_kind}")
        trunk = load_trunk(v.backbone).to(dev)
        if trunk_kind == "finetuned":
            finetune(trunk, v.taps, blobs, tr, y, len(names), small, dev, args.ft_epochs)
        full, tri = extract(trunk, v.taps, blobs, allrows, small, dev)
        if args.gate2:
            gate2(args, cfg, names, full, tri, tr, ev, y, cams, t, order, val, dev)
            return
        for op, X in (("triage", tri), ("full", full)):
            variants = ["none"] if op == "triage" else list(C.VARIANTS)
            for variant in variants:
                for seed in range(args.seeds):
                    model = C.train_context(X[tr], y[tr], cams[tr], t[tr], order[tr], variant,
                                            len(names), seed=seed, device=dev)
                    runs = [("as_trained",) + C.VARIANTS[variant]]
                    if variant == "bi":
                        runs += [(n, C.HORIZON_S, f) for n, f in WAITS]
                    for tag, past, future in runs:
                        pred = C.predict(model, X[ev], cams[ev], t[ev], order[ev], past, future,
                                         device=dev).argmax(1)
                        pc = per_camera(pred, y[ev], val[ev], cams[ev])
                        for c, n, a, vs in pc:
                            rows.append(dict(corpus=cfg.data.corpus, trunk=trunk_kind, op=op, variant=variant,
                                             window=tag, seed=seed, camera=c, n=n, acc=a, value=vs))
                        summary.append(dict(corpus=cfg.data.corpus, trunk=trunk_kind, op=op, variant=variant,
                                            window=tag, seed=seed, acc_micro=float((pred == y[ev]).mean()),
                                            acc_cam=float(np.mean([p[2] for p in pc])),
                                            value_cam=float(np.mean([p[3] for p in pc]))))
                    s = summary[-len(runs)]
                    print(f"  {op:<6} {variant:<13} seed {seed}: acc {s['acc_micro']:.3f}  "
                          f"camera value {s['value_cam']:.3f}", flush=True)
        write_csv(f"{cfg.results_dir}/tables/context_dev_cameras.csv", rows)
        write_csv(f"{cfg.results_dir}/tables/context_dev_summary.csv", summary)
    print("extra MACs of the context read:", C.context_macs(full.shape[1], len(names)))


if __name__ == "__main__":
    main()
