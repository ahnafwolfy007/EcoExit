#!/usr/bin/env python
"""Train the exit heads and the triage detector, calibrate them, record outcomes.

    python pipeline/02_train_exits.py --corpus cct20

Roles are location-disjoint: heads are fit on train cameras, temperatures and
refinement gains on calib cameras, and outcomes are recorded for eval cameras,
which are the simulated deployments and are used for nothing else.

Writes outputs/<corpus>/artifacts/outcomes.npz and heads.pt, plus
results/tables/accuracy_grid.csv and refinement_gain.csv.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sunsched.cli import banner, base_parser, load_cfg, write_csv
from sunsched.data.corpus import load_corpus
from sunsched.eval.metrics import macro_f1
from sunsched.vision.cost import head_macs
from sunsched.vision.heads import (expected_calibration_error, fit_temperature, softmax,
                                   train_head)

ROLES = ("train", "calib", "eval")


def features(feat_dir, role, ri, ti):
    return np.load(f"{feat_dir}/{role}_r{ri}_t{ti}.npy").astype(np.float32)


def frame_values(labels, names, task):
    v = np.full(len(labels), task.value_animal)
    for i, name in enumerate(names):
        if name == task.empty_class:
            v[labels == i] = task.value_empty
        elif name in task.vehicle_classes:
            v[labels == i] = task.value_vehicle
    return v


def isotonic_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: the closest non-decreasing sequence."""
    blocks = []                                     # [mean, weight, count]
    for yi, wi in zip(y.astype(float), np.maximum(w.astype(float), 1e-9)):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    return np.concatenate([[m] * c for m, _, c in blocks])


def gain_table(p_animal, correct_tri, correct_ref, values, edges, prior):
    """Expected value gained by refining, per bin of triage animal probability.

    Shrunk towards the overall mean where a bin has few calibration frames, made
    non-decreasing in p_animal (a frame the detector thinks more likely holds an
    animal should never be worth less to refine), and clipped at zero.
    """
    delta = values * (correct_ref.astype(float) - correct_tri.astype(float))
    overall = float(delta.mean()) if len(delta) else 0.0
    nb = len(edges) - 1
    b = np.clip(np.searchsorted(edges, p_animal, side="right") - 1, 0, nb - 1)
    counts = np.bincount(b, minlength=nb)
    sums = np.bincount(b, weights=delta, minlength=nb)
    raw = (sums + prior * overall) / (counts + prior)
    return np.maximum(isotonic_increasing(raw, counts + prior), 0.0), raw, counts


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg = load_cfg(args)
    v, task = cfg.vision, cfg.task
    feat_dir = f"{cfg.artifacts_dir}/features"
    with open(f"{feat_dir}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    names = meta["class_names"]
    nc = len(names)
    empty_idx = names.index(task.empty_class)

    banner("1. labels")
    corpus = load_corpus(cfg, quick=args.quick)
    recs, y = {}, {}
    for role in ROLES:
        by_id = {r.id: r for r in corpus.roles[role]}
        with open(f"{feat_dir}/{role}_ids.json", encoding="utf-8") as f:
            recs[role] = [by_id[i] for i in json.load(f)]
        y[role] = np.array([names.index(r.label) for r in recs[role]], dtype=np.int64)
        print(f"  {role:<6} {len(y[role]):6,} frames, {np.mean(y[role] == empty_idx) * 100:5.1f}% empty")

    banner("2. species heads at every (resolution, exit)")
    outs, heads, grid = {}, {}, []
    for ri in range(len(v.resolutions)):
        for ti in range(len(v.taps)):
            Xtr = features(feat_dir, "train", ri, ti)
            print(f"  input {meta['input_sizes'][ri]}, exit {ti} (tap {v.taps[ti]}, d={Xtr.shape[1]})")
            th = train_head(Xtr, y["train"], nc, v.head_hidden, v.dropout, v.epochs, v.lr,
                            v.weight_decay, v.seed, log=print)
            lg_cal = th.logits(features(feat_dir, "calib", ri, ti))
            ece_before = expected_calibration_error(softmax(lg_cal).max(1), softmax(lg_cal).argmax(1) == y["calib"])
            th.temperature = fit_temperature(lg_cal, y["calib"])
            p_cal, p_ev = th.probs(features(feat_dir, "calib", ri, ti)), th.probs(features(feat_dir, "eval", ri, ti))
            macs = meta["trunk_macs"][str(ri)][str(v.taps[ti])] + head_macs(Xtr.shape[1], v.head_hidden, nc)
            outs[(ri, ti)] = dict(pred_ev=p_ev.argmax(1), conf_ev=p_ev.max(1),
                                  correct_cal=p_cal.argmax(1) == y["calib"], macs=macs)
            heads[f"species_{ri}_{ti}"] = th
            row = dict(resolution=meta["input_sizes"][ri][0], exit=ti, tap=v.taps[ti],
                       mmacs=round(macs / 1e6, 2), temperature=round(th.temperature, 3),
                       acc_calib=float(np.mean(p_cal.argmax(1) == y["calib"])),
                       acc_eval=float(np.mean(p_ev.argmax(1) == y["eval"])),
                       macro_f1_eval=macro_f1(y["eval"], p_ev.argmax(1)),
                       animal_vs_empty_acc_eval=float(np.mean((p_ev.argmax(1) == empty_idx) == (y["eval"] == empty_idx))),
                       top1_conf_p10=float(np.quantile(p_ev.max(1), 0.1)),
                       top1_conf_p90=float(np.quantile(p_ev.max(1), 0.9)),
                       ece_calib_before=ece_before,
                       ece_calib_after=expected_calibration_error(p_cal.max(1), p_cal.argmax(1) == y["calib"]))
            grid.append(row)
            print(f"    eval acc {row['acc_eval']:.3f}  macro-F1 {row['macro_f1_eval']:.3f}  "
                  f"top-1 conf p10-p90 {row['top1_conf_p10']:.3f}-{row['top1_conf_p90']:.3f}")

    banner("3. triage detector (animal vs empty)")
    tri = tuple(v.triage_op)
    Xtr = features(feat_dir, "train", *tri)
    is_animal = {role: (y[role] != empty_idx).astype(np.int64) for role in ROLES}
    det = train_head(Xtr, is_animal["train"], 2, v.head_hidden, v.dropout, v.epochs, v.lr,
                     v.weight_decay, v.seed, log=print)
    det.temperature = fit_temperature(det.logits(features(feat_dir, "calib", *tri)), is_animal["calib"])
    pa_cal = det.probs(features(feat_dir, "calib", *tri))[:, 1]
    pa_ev = det.probs(features(feat_dir, "eval", *tri))[:, 1]
    heads["detector"] = det
    det_acc = float(np.mean((pa_ev >= 0.5) == is_animal["eval"]))
    print(f"  eval accuracy {det_acc:.3f}; p_animal p10-p90 on eval "
          f"{np.quantile(pa_ev, 0.1):.3f}-{np.quantile(pa_ev, 0.9):.3f} "
          f"(an informative signal spreads across [0, 1])")
    det_macs = meta["trunk_macs"][str(tri[0])][str(v.taps[tri[1]])] + head_macs(Xtr.shape[1], v.head_hidden, 2)
    grid.append(dict(resolution=meta["input_sizes"][tri[0]][0], exit=tri[1], tap="detector",
                     mmacs=round(det_macs / 1e6, 2), temperature=round(det.temperature, 3),
                     acc_eval=det_acc, animal_vs_empty_acc_eval=det_acc,
                     ece_calib_after=expected_calibration_error(np.maximum(pa_cal, 1 - pa_cal),
                                                                (pa_cal >= 0.5) == is_animal["calib"])))
    write_csv(f"{cfg.results_dir}/tables/accuracy_grid.csv", grid)

    banner("4. operating points and refinement gains")
    ops = {"triage": tri, **{k: tuple(x) for k, x in v.refine_ops.items()}}
    values_cal = frame_values(y["calib"], names, task)
    edges = np.unique(np.quantile(pa_cal, np.linspace(0, 1, v.n_conf_bins + 1)))
    edges[0], edges[-1] = 0.0, 1.0 + 1e-9
    save = dict(class_names=np.array(names), ops=np.array(list(ops)),
                eval_ids=np.array([r.id for r in recs["eval"]]), eval_labels=y["eval"],
                eval_values=frame_values(y["eval"], names, task),
                eval_locations=np.array([r.location for r in recs["eval"]]),
                eval_timestamps=np.array([r.timestamp.isoformat() for r in recs["eval"]]),
                p_animal_triage=pa_ev.astype(np.float32), gain_edges=edges)
    gain_rows = []
    for op, key in ops.items():
        o = outs[key]
        save[f"pred_{op}"] = o["pred_ev"]
        save[f"conf_{op}"] = o["conf_ev"].astype(np.float32)
        # Triage also runs the detector head; its cost belongs to the triage op.
        save[f"macs_{op}"] = np.int64(o["macs"] + (det_macs - meta["trunk_macs"][str(tri[0])][str(v.taps[tri[1]])]
                                                   if op == "triage" else 0))
        print(f"  {op:<7} input idx {key[0]}, exit {key[1]}: {o['macs'] / 1e6:7.1f} MMACs, "
              f"eval acc {np.mean(o['pred_ev'] == y['eval']):.3f}")
        if op == "triage":
            continue
        table, raw, counts = gain_table(pa_cal, outs[tri]["correct_cal"], o["correct_cal"],
                                        values_cal, edges, v.gain_prior_count)
        save[f"gain_{op}"] = table
        for i in range(len(table)):
            gain_rows.append(dict(op=op, p_animal_lo=round(edges[i], 4), p_animal_hi=round(edges[i + 1], 4),
                                  raw_gain=round(raw[i], 5), expected_gain=round(table[i], 5),
                                  calib_frames=int(counts[i])))
        print(f"    gain by p_animal bin (low -> high): {np.round(table, 3).tolist()}")
    write_csv(f"{cfg.results_dir}/tables/refinement_gain.csv", gain_rows)
    np.savez_compressed(f"{cfg.artifacts_dir}/outcomes.npz", **save)
    torch.save({k: dict(state=h.head.state_dict(), mean=h.mean, std=h.std, temperature=h.temperature)
                for k, h in heads.items()}, f"{cfg.artifacts_dir}/heads.pt")
    print(f"  wrote {cfg.artifacts_dir}/outcomes.npz and heads.pt")


if __name__ == "__main__":
    main()
