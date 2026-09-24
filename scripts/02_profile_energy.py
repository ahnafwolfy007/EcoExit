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

    # -- 2. leakage-free pools ----------------------------------------------
    # The original version of this script had three leaks, all fixed here:
    #
    #   (a) Temperature scaling was fitted on the test logits.
    #   (b) The accuracy grid handed to the controller as its value function
    #       was the test grid, and the simulation was then scored on that same
    #       set -- so the controller was tuned on its own evaluation data.
    #   (c) Per-image outcomes for the simulation came from that same pool.
    #
    # The fix: fit temperature and build the controller's value grid on
    # VALIDATION, and split the test set into disjoint calibration and
    # evaluation pools. The simulator only ever sees the evaluation pool.
    from ecoexit.data.cifar import make_loaders
    _, val_loader, test_loader, test_ds = make_loaders(cfg.train, root=args.data_root)
    n_classes = int(ckpt["n_classes"])

    def forward_logits(loader, n):
        """logits[ri][e] -> (n, n_classes) and the matching labels."""
        L = {ri: {e: np.zeros((n, n_classes), dtype=np.float32) for e in range(n_exits)}
             for ri in range(len(resolutions))}
        labels = np.zeros(n, dtype=np.int64)
        with torch.no_grad():
            off = 0
            for x, y in loader:
                bs = y.size(0)
                for ri in range(len(resolutions)):
                    set_resolution_idx(ri)
                    outs = model(model.resize_for(x, ri), upto=n_exits - 1)
                    for e, lg in enumerate(outs):
                        L[ri][e][off:off + bs] = lg.numpy()
                labels[off:off + bs] = y.numpy()
                off += bs
        return L, labels

    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    val_logits, val_labels = forward_logits(val_loader, n_val)
    test_logits, test_labels = forward_logits(test_loader, n_test)
    print(f"[energy] forward passes done: {n_val} val, {n_test} test images "
          f"x {len(resolutions)} resolutions x {n_exits} exits")

    # Temperature per (resolution, exit), fitted on VALIDATION only.
    temps = np.ones((len(resolutions), n_exits), dtype=np.float32)
    for ri in range(len(resolutions)):
        for e in range(n_exits):
            temps[ri, e] = fit_temperature(val_logits[ri][e], val_labels)
    print("[energy] fitted temperatures (val):\n" + str(np.round(temps, 3)))

    def scored(logits_by_res, labels, apply_temp=True):
        correct = np.zeros((len(resolutions), n_exits, len(labels)), dtype=bool)
        conf = np.zeros((len(resolutions), n_exits, len(labels)), dtype=np.float32)
        for ri in range(len(resolutions)):
            for e in range(n_exits):
                lg = torch.tensor(logits_by_res[ri][e])
                if apply_temp:
                    lg = lg / float(temps[ri, e])
                p = F.softmax(lg, dim=1).numpy()
                correct[ri, e] = (p.argmax(1) == labels)
                conf[ri, e] = p.max(1)
        return correct, conf

    # The controller's value function: validation accuracy, never test.
    correct_val, conf_val = scored(val_logits, val_labels)
    acc_grid_val = correct_val.mean(axis=2)

    # Disjoint calibration / evaluation pools over the test set.
    rng = np.random.default_rng(cfg.train.seed)
    perm = rng.permutation(n_test)
    calib_pool = np.sort(perm[: n_test // 2])
    eval_pool = np.sort(perm[n_test // 2:])
    assert len(np.intersect1d(calib_pool, eval_pool)) == 0

    correct_test, conf_test = scored(test_logits, test_labels)
    correct = correct_test[:, :, eval_pool]
    conf = conf_test[:, :, eval_pool]
    acc_grid = correct.mean(axis=2)          # reported, never given to the controller

    print(f"[energy] pools: {len(calib_pool)} calibration, {len(eval_pool)} evaluation "
          f"(disjoint). Controller sees the validation grid only.")
    np.savez(f"{args.artifacts}/per_image_outcomes.npz",
             correct=correct, conf=conf, resolutions=np.array(resolutions),
             eval_pool=eval_pool, calib_pool=calib_pool,
             labels=test_labels[eval_pool])
    print(f"[energy] wrote {args.artifacts}/per_image_outcomes.npz")

    # -- 3. calibration curves, measured on the calibration pool -------------
    # These are temperature-scaled, and the temperature is the same one the
    # simulator's stored confidences carry, so the report's claim that
    # calibration makes the confidence stopping rule meaningful is now true.
    # Previously the report asserted it while the simulator read raw softmax.
    curves = {}
    ri_native = len(resolutions) - 1
    for e in range(n_exits):
        c = conf_test[ri_native, e][calib_pool]
        cor = correct_test[ri_native, e][calib_pool]
        curves[e] = reliability_curve(c, cor)
        ece = float(np.sum(curves[e][2] / max(curves[e][2].sum(), 1)
                           * np.abs(curves[e][0] - curves[e][1])))
        print(f"[energy] exit {e}: T={temps[ri_native, e]:.2f}  ECE={ece:.4f}")

    np.savez(f"{args.artifacts}/confidence_calibration.npz",
             temps=temps, temps_native=temps[ri_native],
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
        ax.set_title(f"Exit {e} (T={temps[ri_native, e]:.2f})")
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, grid, name in zip(axes, [acc_grid_val, acc_grid],
                              ["Validation (controller's value function)",
                               "Evaluation pool (reported only)"]):
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=max(grid.max(), 1e-6) * 1.05,
                       aspect="auto")
        ax.set_xticks(range(n_exits)); ax.set_xticklabels([f"exit {e}" for e in range(n_exits)])
        ax.set_yticks(range(len(resolutions)))
        ax.set_yticklabels([f"{r}px" for r in resolutions])
        for ri in range(len(resolutions)):
            for e in range(n_exits):
                ax.text(e, ri, f"{grid[ri, e]:.3f}", ha="center", va="center",
                        color="white" if grid[ri, e] < grid.max() * 0.6 else "black",
                        fontsize=9)
        ax.set_title(name, fontsize=10)
        fig.colorbar(im, ax=ax, label="top-1 accuracy")
    fig.suptitle("Accuracy: resolution x exit depth, on disjoint pools")
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/accuracy_grid.png", dpi=150)
    plt.close(fig)
    print(f"[energy] wrote {args.out}/figures/accuracy_grid.png")
    print(f"[energy] val grid (controller):\n{np.round(acc_grid_val, 4)}")
    print(f"[energy] eval grid (reported):\n{np.round(acc_grid, 4)}")

    # Non-monotonicity in the grid indicates an under-converged model, not a
    # finding. Flag it here so it cannot quietly become a claim downstream.
    for ri in range(len(resolutions)):
        if np.any(np.diff(acc_grid_val[ri]) < -1e-3):
            print(f"[energy] WARNING: accuracy is non-monotone in exit depth at "
                  f"{resolutions[ri]}px -- train longer before claiming anything "
                  f"about the accuracy/energy frontier.")

    np.savez(f"{args.artifacts}/final_accuracy_grid.npz",
             acc_grid=acc_grid, acc_grid_val=acc_grid_val,
             resolutions=np.array(resolutions))
    print("[energy] done.")


if __name__ == "__main__":
    main()
