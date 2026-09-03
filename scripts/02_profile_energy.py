#!/usr/bin/env python
"""Profile the trained backbone's real energy, compare the decomposed model
against a MACs-only proxy (Contribution D / Hole H1), and precompute the
per-test-image correctness+confidence table every simulation needs.

Usage:
    python scripts/02_profile_energy.py

Produces (under results/ and artifacts/):
    tables/energy_table.csv          -- measured MACs/bytes/time per (r,k) + both $ models
    tables/mac_rank_mismatch.json    -- Contribution D's headline numbers
    figures/energy_mismatch.png      -- decomposed vs MACs-only scatter, mismatched pairs marked
    figures/accuracy_grid.png        -- accuracy heatmap over (resolution, exit)
    artifacts/per_image_outcomes.npz -- correct[r][k][img], conf[r][k][img] for the test set
    artifacts/confidence_calibration.npz -- per-exit temperature + reliability curve
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecoexit.config import Config
from ecoexit.data.cifar import get_datasets
from ecoexit.models.elastic import ElasticNet, set_resolution_idx
from ecoexit.energy.model import (
    profile_model, DecomposedModel, MacsOnlyProxy, build_energy_table, mac_rank_mismatch,
)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """One-parameter temperature scaling (Guo et al. 2017) fit by grid search
    -- keeps this dependency-free instead of pulling in an optimiser."""
    logits_t = torch.tensor(logits)
    labels_t = torch.tensor(labels)
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.5, 3.0, 26):
        nll = F.cross_entropy(logits_t / T, labels_t).item()
        if nll < best_nll:
            best_nll, best_T = nll, T
    return float(best_T)


def reliability_curve(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    accs, confs, counts = [], [], []
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < n_bins - 1 else conf <= edges[i + 1])
        if m.sum() > 0:
            accs.append(correct[m].mean())
            confs.append(conf[m].mean())
            counts.append(int(m.sum()))
        else:
            accs.append(np.nan)
            confs.append((edges[i] + edges[i + 1]) / 2)
            counts.append(0)
    return np.array(confs), np.array(accs), np.array(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--n-timing-runs", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(f"{args.out}/tables", exist_ok=True)
    os.makedirs(f"{args.out}/figures", exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)

    ckpt = torch.load(f"{args.artifacts}/backbone.pt", map_location="cpu", weights_only=False)
    cfg = Config()
    model = ElasticNet(ckpt["resolutions"], ckpt["n_exits"], ckpt["base_width"],
                        ckpt["blocks_per_stage"], ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    resolutions = tuple(int(r) for r in ckpt["resolutions"])
    n_exits = int(ckpt["n_exits"])
    print(f"[energy] loaded backbone: resolutions={resolutions} n_exits={n_exits}")

    # -- 1. measured MACs / bytes / time -----------------------------------
    points = profile_model(model, resolutions, n_exits, set_resolution_idx,
                            n_timing_runs=args.n_timing_runs)
    decomposed = DecomposedModel(cfg.energy)
    macs_only = MacsOnlyProxy()
    dt = build_energy_table(points, decomposed)
    mt = build_energy_table(points, macs_only)
    mismatch = mac_rank_mismatch(dt, mt)
    print(f"[energy] MACs-only vs decomposed: {mismatch['n_mismatch']}/{mismatch['n_pairs']} pairs "
          f"disagree on ranking ({100*mismatch['mismatch_rate']:.1f}%); "
          f"cross-resolution mismatch rate = {100*mismatch['cross_res_mismatch_rate']:.1f}%; "
          f"Spearman rho = {mismatch['spearman_rho']:.3f}")

    with open(f"{args.out}/tables/mac_rank_mismatch.json", "w") as f:
        json.dump(mismatch, f, indent=2)

    with open(f"{args.out}/tables/energy_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resolution", "exit", "macs", "act_bytes", "weight_bytes", "time_ms",
                    "energy_decomposed_j", "energy_macs_only_j"])
        for p in points:
            w.writerow([p.resolution, p.exit_idx, p.macs, p.act_bytes, p.weight_bytes,
                        p.time_s * 1e3, dt[(p.res_idx, p.exit_idx)], mt[(p.res_idx, p.exit_idx)]])
    print(f"[energy] wrote {args.out}/tables/energy_table.csv")

    # -- figure: decomposed vs MACs-only ranking scatter --------------------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    d_vals = np.array([dt[(p.res_idx, p.exit_idx)] for p in points])
    m_vals = np.array([mt[(p.res_idx, p.exit_idx)] for p in points])
    colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))
    for p, dv, mv in zip(points, d_vals, m_vals):
        ax.scatter(mv * 1e3, dv * 1e3, color=colors[p.res_idx], s=70,
                   edgecolor="k", linewidth=0.5, zorder=3)
        ax.annotate(f"r{p.resolution}k{p.exit_idx}", (mv * 1e3, dv * 1e3),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    lims = [0, max(d_vals.max(), m_vals.max()) * 1.15 * 1e3]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="agreement line")
    ax.set_xlabel("MACs-only proxy energy (mJ)")
    ax.set_ylabel("Decomposed (measured) energy (mJ)")
    ax.set_title(f"Energy proxy disagreement: {mismatch['n_mismatch']}/{mismatch['n_pairs']} "
                 f"pairs mis-ranked ({100*mismatch['mismatch_rate']:.0f}%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/energy_mismatch.png", dpi=150)
    plt.close(fig)
    print(f"[energy] wrote {args.out}/figures/energy_mismatch.png")

    # -- 2. per-test-image correctness + confidence at every (r,k) ---------
    _, test_ds = get_datasets(args.data_root, download=True)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=500, shuffle=False)
    n_test = len(test_ds)
    correct = np.zeros((len(resolutions), n_exits, n_test), dtype=bool)
    conf = np.zeros((len(resolutions), n_exits, n_test), dtype=np.float32)
    raw_logits = {e: [] for e in range(n_exits)}
    raw_labels = []

    with torch.no_grad():
        offset = 0
        for x, y in loader:
            bs = y.size(0)
            for ri in range(len(resolutions)):
                set_resolution_idx(ri)
                xr = model.resize_for(x, ri)
                outs = model(xr, upto=n_exits - 1)
                for e, logits in enumerate(outs):
                    probs = F.softmax(logits, dim=1)
                    c, pred = probs.max(1)
                    correct[ri, e, offset:offset + bs] = (pred == y).numpy()
                    conf[ri, e, offset:offset + bs] = c.numpy()
                    if ri == len(resolutions) - 1:
                        raw_logits[e].append(logits.numpy())
            raw_labels.append(y.numpy())
            offset += bs
    raw_labels = np.concatenate(raw_labels)
    print(f"[energy] precomputed correctness/confidence for {n_test} test images "
          f"x {len(resolutions)} resolutions x {n_exits} exits")

    np.savez(f"{args.artifacts}/per_image_outcomes.npz",
              correct=correct, conf=conf, resolutions=np.array(resolutions))
    print(f"[energy] wrote {args.artifacts}/per_image_outcomes.npz")

    # -- 3. confidence calibration (temperature scaling) at native resolution
    temps, curves = {}, {}
    for e in range(n_exits):
        logits_e = np.concatenate(raw_logits[e], axis=0)
        T = fit_temperature(logits_e, raw_labels)
        probs = F.softmax(torch.tensor(logits_e) / T, dim=1).numpy()
        c = probs.max(1)
        pred = probs.argmax(1)
        cor = (pred == raw_labels)
        temps[e] = T
        curves[e] = reliability_curve(c, cor)
        print(f"[energy] exit {e}: fitted temperature T={T:.2f}")

    np.savez(f"{args.artifacts}/confidence_calibration.npz",
              temps=np.array([temps[e] for e in range(n_exits)]),
              **{f"conf_bins_e{e}": curves[e][0] for e in range(n_exits)},
              **{f"acc_bins_e{e}": curves[e][1] for e in range(n_exits)},
              **{f"count_bins_e{e}": curves[e][2] for e in range(n_exits)})

    fig, axes = plt.subplots(1, n_exits, figsize=(4 * n_exits, 4), sharey=True)
    if n_exits == 1:
        axes = [axes]
    for e, ax in enumerate(axes):
        cbins, abins, cnt = curves[e]
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.plot(cbins, abins, "o-", color="#9A6508")
        ax.set_title(f"Exit {e} (T={temps[e]:.2f})")
        ax.set_xlabel("Confidence")
        if e == 0:
            ax.set_ylabel("Empirical accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Confidence calibration per exit (post temperature-scaling)")
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/calibration_curves.png", dpi=150)
    plt.close(fig)
    print(f"[energy] wrote {args.out}/figures/calibration_curves.png")

    # -- accuracy grid heatmap ----------------------------------------------
    acc_grid = correct.mean(axis=2)  # (n_res, n_exit)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(acc_grid, cmap="viridis", vmin=0, vmax=acc_grid.max() * 1.05, aspect="auto")
    ax.set_xticks(range(n_exits)); ax.set_xticklabels([f"exit {e}" for e in range(n_exits)])
    ax.set_yticks(range(len(resolutions))); ax.set_yticklabels([f"{r}px" for r in resolutions])
    for ri in range(len(resolutions)):
        for e in range(n_exits):
            ax.text(e, ri, f"{acc_grid[ri,e]:.3f}", ha="center", va="center",
                    color="white" if acc_grid[ri,e] < acc_grid.max()*0.6 else "black", fontsize=9)
    ax.set_title("Test accuracy: resolution x exit depth")
    fig.colorbar(im, ax=ax, label="top-1 accuracy")
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/accuracy_grid.png", dpi=150)
    plt.close(fig)
    print(f"[energy] wrote {args.out}/figures/accuracy_grid.png  (grid=\n{acc_grid}\n)")

    np.savez(f"{args.artifacts}/final_accuracy_grid.npz", acc_grid=acc_grid, resolutions=np.array(resolutions))
    print("[energy] done.")


if __name__ == "__main__":
    main()
