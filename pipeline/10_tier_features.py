#!/usr/bin/env python
"""Fine-tune the tiers on a corpus and cache features for the head experiments.

    python pipeline/10_tier_features.py --corpus cct20 --device cuda --models b_mnv3l b_mnv4m b_mnv4h a_mnv4s

Development corpora only (PLAN_V2 gates D1-D4). Tier B models are trained first;
tier A is trained twice, plainly and distilled from the best tier B of this
corpus by training-camera calibration loss (no evaluation data is used).
"""
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sunsched.cli import banner, base_parser, load_cfg, write_csv
from sunsched.data import cct20, source
from sunsched.vision import device as vision_device
from sunsched.vision import tiers as TR
from sunsched.vision.backbone import input_size


def load_blobs(cfg, quick, h, w):
    splits, names = source.load_splits(cfg, quick=quick)
    recs = [(s, r) for s in source.used_splits(cfg) for r in splits[s]]
    where = {r.file_name: i for i, (_, r) in enumerate(recs)}
    blobs = [None] * len(recs)
    for name, jpeg in source.image_source(cfg, where.keys(), ""):
        try:
            b = io.BytesIO()
            cct20.decode(jpeg, h, w).save(b, "JPEG", quality=92)
            blobs[where[name]] = b.getvalue()
        except Exception:
            continue
    keep = [i for i, b in enumerate(blobs) if b is not None]
    return [recs[i] for i in keep], [blobs[i] for i in keep], names, len(recs) - len(keep)


def camera_scores(pred, y, val, cams):
    out = []
    for c in np.unique(cams):
        m = cams == c
        out.append((c, int(m.sum()), float((pred[m] == y[m]).mean()), float((val[m] * (pred[m] == y[m])).sum() / val[m].sum())))
    return out


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--models", nargs="+", default=["b_mnv3l", "b_mnv4m", "b_mnv4h", "a_mnv4s"])
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    cfg = load_cfg(args)
    dev = vision_device.resolve(cfg.vision.device)
    a = cfg.vision.aspect
    banner(f"1. images ({cfg.data.corpus})")
    recs, blobs, names, missing = load_blobs(cfg, args.quick, *input_size(320, a))
    role = np.array(["train" if s in cfg.data.train_splits else "calib" if s == cfg.data.calib_split else "eval"
                     for s, _ in recs])
    y = np.array([names.index(r.label) for _, r in recs], dtype=np.int64)
    cams = np.array([str(r.location) for _, r in recs])
    t = np.array([r.timestamp.timestamp() for _, r in recs])
    order = np.array([r.frame_num for _, r in recs])
    val = np.array([cfg.task.value_empty if names[k] == cfg.task.empty_class else
                    cfg.task.value_vehicle if names[k] in cfg.task.vehicle_classes else cfg.task.value_animal for k in y])
    tr, ca, ev = (np.where(role == k)[0] for k in ("train", "calib", "eval"))
    print(f"  train {len(tr)} ({len(set(cams[tr]))} cameras), calib {len(ca)}, eval {len(ev)} "
          f"({len(set(cams[ev]))}); {len(names)} classes; {missing} images missing; {sum(map(len, blobs)) / 1e9:.2f} GB")
    out = f"{args.out}/tier_features_{cfg.data.corpus}"
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/meta.npz", role=role, y=y, cams=cams, t=t, order=order, val=val, names=np.array(names),
             ids=np.array([r.id for _, r in recs]))

    rows, teachers = [], {}
    everything = np.arange(len(recs))

    def one(tag, name, teacher=None):
        banner(f"2. {tag}")
        timm_name, height = TR.TIERS[name]
        size = input_size(height, a)
        trunk = TR.Trunk(timm_name).to(dev)
        macs = TR.count_macs(trunk, *size)
        params = sum(p.numel() for p in trunk.parameters())
        print(f"  {timm_name} at {size}: {macs / 1e6:.0f} MMAC, {params / 1e6:.2f} M parameters", flush=True)
        t0 = time.time()
        trunk, head = TR.finetune(trunk, size, blobs, tr, y, len(names), dev, epochs=args.epochs, teacher=teacher)
        X, Z = TR.extract(trunk, head, size, blobs, everything, dev)
        np.save(f"{out}/X_{tag}.npy", X)
        np.save(f"{out}/Z_{tag}.npy", Z)
        torch.save(dict(trunk={k: v.half().cpu() if v.is_floating_point() else v.cpu() for k, v in trunk.state_dict().items()},
                        head=head.state_dict(), model=timm_name, size=size), f"{out}/{tag}.pt")
        sc = camera_scores(Z[ev].argmax(1), y[ev], val[ev], cams[ev])
        row = dict(corpus=cfg.data.corpus, tag=tag, model=timm_name, height=size[0], width=size[1], mmacs=macs / 1e6,
                   params_m=params / 1e6, acc=float((Z[ev].argmax(1) == y[ev]).mean()),
                   cam_acc=float(np.mean([s[2] for s in sc])), cam_value=float(np.mean([s[3] for s in sc])),
                   calib_acc=float((Z[ca].argmax(1) == y[ca]).mean()), minutes=(time.time() - t0) / 60)
        rows.append(row)
        write_csv(f"{args.out}/results/tables/tier_summary_{cfg.data.corpus}.csv", rows)
        print(f"  eval acc {row['acc']:.3f}, camera value {row['cam_value']:.3f}, calib acc {row['calib_acc']:.3f}, "
              f"{row['minutes']:.0f} min", flush=True)
        return Z, row

    for name in [m for m in args.models if m.startswith("b_")]:
        teachers[name] = one(name, name)
    for name in [m for m in args.models if m.startswith("a_")]:
        one(name, name)
        if teachers:
            best = max(teachers, key=lambda k: teachers[k][1]["calib_acc"])
            print(f"  distilling from {best} (best calibration accuracy)")
            one(f"{name}_kd", name, teachers[best][0])


if __name__ == "__main__":
    main()
