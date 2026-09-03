#!/usr/bin/env python
"""Train the elastic (multi-resolution x multi-exit) backbone on CIFAR-100.

Usage:
    python scripts/01_train_backbone.py            # full run
    python scripts/01_train_backbone.py --quick     # ~5-10 min laptop smoke test

What this produces (all under results/ and artifacts/):
    logs/train_log.csv        -- per-epoch train loss, per (res,exit) val accuracy
    figures/training_curves.png
    artifacts/backbone.pt     -- the trained checkpoint
    artifacts/test_accuracy_grid.npz, conf_records_exit*.npy

(The final TEST-set accuracy grid figure is produced by scripts/02, which is
the more authoritative number -- this script's own figure is the training
trajectory, on the validation split.)
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecoexit.config import Config, quick as quick_cfg
from ecoexit.data.cifar import make_loaders
from ecoexit.models.elastic import (
    ElasticNet, set_resolution_idx, multi_exit_loss, sandwich_indices,
)


def evaluate(model, loader, resolutions, n_exits, device):
    """Returns acc[res_idx][exit_idx] and, for the largest resolution, a list
    of (confidence, correct) pairs per exit for later calibration."""
    model.eval()
    n_res = len(resolutions)
    correct = np.zeros((n_res, n_exits))
    total = 0
    conf_records = {e: [] for e in range(n_exits)}  # at native/largest res only
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += y.size(0)
            for ri in range(n_res):
                set_resolution_idx(ri)
                xr = model.resize_for(x, ri)
                outs = model(xr, upto=n_exits - 1)
                for e, logits in enumerate(outs):
                    pred = logits.argmax(1)
                    correct[ri, e] += (pred == y).sum().item()
                    if ri == n_res - 1:
                        probs = F.softmax(logits, dim=1)
                        conf, _ = probs.max(1)
                        is_correct = (pred == y).cpu().numpy()
                        conf_records[e].append(np.stack([conf.cpu().numpy(), is_correct], axis=1))
    acc = correct / max(total, 1)
    conf_records = {e: np.concatenate(v, axis=0) for e, v in conf_records.items()}
    return acc, conf_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast laptop smoke-test preset")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    args = ap.parse_args()

    cfg = Config()
    if args.quick:
        cfg = quick_cfg(cfg)
    if args.epochs:
        cfg.train.epochs = args.epochs

    os.makedirs(f"{args.out}/logs", exist_ok=True)
    os.makedirs(f"{args.out}/figures", exist_ok=True)
    os.makedirs(args.artifacts, exist_ok=True)

    torch.set_num_threads(os.cpu_count() or 4)
    torch.manual_seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} quick={args.quick} epochs={cfg.train.epochs}")

    train_loader, val_loader, test_loader, _ = make_loaders(cfg.train, root=args.data_root)
    print(f"[train] train batches={len(train_loader)} val={len(val_loader)} test={len(test_loader)}")

    model = ElasticNet(cfg.model.resolutions, cfg.model.n_exits, cfg.model.base_width,
                        cfg.model.blocks_per_stage, cfg.model.n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] elastic backbone params={n_params:,} "
          f"resolutions={cfg.model.resolutions} exits={cfg.model.n_exits}")

    opt = torch.optim.SGD(model.parameters(), lr=cfg.train.lr,
                           momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)

    n_res = len(cfg.model.resolutions)
    log_rows = []
    t_start = time.time()

    for epoch in range(cfg.train.epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            res_idxs = sandwich_indices(n_res, cfg.train.sandwich_n_random, rng)
            loss_total = 0.0
            for ri in res_idxs:
                set_resolution_idx(ri)
                xr = model.resize_for(x, ri)
                outs = model(xr, upto=cfg.model.n_exits - 1)
                loss_total = loss_total + multi_exit_loss(outs, y, cfg.model.exit_loss_weights)
            loss_total.backward()
            opt.step()
            running_loss += float(loss_total.detach()) / max(len(res_idxs), 1)
            n_batches += 1
        sched.step()

        val_acc, _ = evaluate(model, val_loader, cfg.model.resolutions, cfg.model.n_exits, device)
        dt = time.time() - t0
        mean_loss = running_loss / max(n_batches, 1)
        best_acc = val_acc.max()
        print(f"[train] epoch {epoch+1:>3}/{cfg.train.epochs}  loss={mean_loss:.4f}  "
              f"val_best_acc={best_acc:.4f}  time={dt:.1f}s")

        row = {"epoch": epoch + 1, "train_loss": mean_loss, "time_s": dt}
        for ri, r in enumerate(cfg.model.resolutions):
            for e in range(cfg.model.n_exits):
                row[f"val_acc_r{r}_e{e}"] = val_acc[ri, e]
        log_rows.append(row)

    total_time = time.time() - t_start
    print(f"[train] finished in {total_time/60:.1f} min")

    # Final evaluation on the held-out TEST set + confidence calibration records
    test_acc, conf_records = evaluate(model, test_loader, cfg.model.resolutions,
                                       cfg.model.n_exits, device)
    print("[train] final TEST accuracy grid (rows=resolution, cols=exit):")
    for ri, r in enumerate(cfg.model.resolutions):
        print(f"    res={r:>3}: " + "  ".join(f"exit{e}={test_acc[ri,e]:.4f}" for e in range(cfg.model.n_exits)))

    # -- persist everything ------------------------------------------------
    log_path = f"{args.out}/logs/train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)
    print(f"[train] wrote {log_path}")

    # -- training curves figure ---------------------------------------------
    epochs_x = [r["epoch"] for r in log_rows]
    fig, (axl, axa) = plt.subplots(1, 2, figsize=(11, 4.2))
    axl.plot(epochs_x, [r["train_loss"] for r in log_rows], color="#9A6508", marker="o", ms=3)
    axl.set_xlabel("epoch"); axl.set_ylabel("train loss (summed sandwich-rule exits)")
    axl.set_title("Training loss")

    cmap = plt.cm.viridis(np.linspace(0, 1, len(cfg.model.resolutions)))
    styles = ["-", "--", ":"]
    for ri, r in enumerate(cfg.model.resolutions):
        for e in range(cfg.model.n_exits):
            key = f"val_acc_r{r}_e{e}"
            if key in log_rows[0]:
                axa.plot(epochs_x, [row[key] for row in log_rows], color=cmap[ri],
                          linestyle=styles[e % len(styles)], lw=1.6,
                          label=f"res={r} exit={e}")
    axa.set_xlabel("epoch"); axa.set_ylabel("validation top-1 accuracy")
    axa.set_title("Validation accuracy, every (resolution, exit)")
    axa.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/training_curves.png", dpi=150)
    plt.close(fig)
    print(f"[train] wrote {args.out}/figures/training_curves.png")

    np.savez(f"{args.artifacts}/test_accuracy_grid.npz",
             resolutions=np.array(cfg.model.resolutions), test_acc=test_acc)

    for e, rec in conf_records.items():
        np.save(f"{args.artifacts}/conf_records_exit{e}.npy", rec)

    ckpt = {
        "state_dict": model.state_dict(),
        "resolutions": cfg.model.resolutions,
        "n_exits": cfg.model.n_exits,
        "base_width": cfg.model.base_width,
        "blocks_per_stage": cfg.model.blocks_per_stage,
        "n_classes": cfg.model.n_classes,
        "test_acc": test_acc,
        "total_train_time_s": total_time,
    }
    torch.save(ckpt, f"{args.artifacts}/backbone.pt")
    print(f"[train] wrote {args.artifacts}/backbone.pt")

    cfg.to_json(f"{args.artifacts}/config_used.json")
    print("[train] done.")


if __name__ == "__main__":
    main()
