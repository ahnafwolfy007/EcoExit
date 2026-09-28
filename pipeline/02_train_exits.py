#!/usr/bin/env python
"""Train the exit heads, calibrate them, and record what the simulator needs.

    python pipeline/02_train_exits.py

Data roles, all location-disjoint:
    train + cis_val + cis_test   10 cameras   fit the heads
    trans_val                     1 camera    fit temperatures and refinement-gain tables
    trans_test                    9 cameras   the simulated deployments; used for nothing else

Writes outputs/artifacts/outcomes.npz and outputs/artifacts/heads.pt, and
outputs/results/tables/accuracy_grid.csv.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sunsched.cli import banner, base_parser, load_cfg, write_csv
from sunsched.data import cct20
from sunsched.eval.metrics import macro_f1
from sunsched.vision import device as vision_device
from sunsched.vision.cost import head_macs
from sunsched.vision.heads import (expected_calibration_error, fit_temperature, softmax,
                                   train_head)


def load_features(feat_dir, split, ri, ti):
    return np.load(f"{feat_dir}/{split}_r{ri}_t{ti}.npy").astype(np.float32)


def frame_values(labels, names, task):
    v = np.full(len(labels), task.value_animal)
    for i, name in enumerate(names):
        if name == task.empty_class:
            v[labels == i] = task.value_empty
        elif name in task.vehicle_classes:
            v[labels == i] = task.value_vehicle
    return v


def gain_table(conf_tri, correct_tri, correct_ref, is_empty_pred, values, edges, prior):
    """Expected value gained by refining, by triage confidence bin and by
    whether triage said "empty", shrunk towards the overall mean where a bin
    has few calibration frames."""
    delta = values * (correct_ref.astype(float) - correct_tri.astype(float))
    overall = float(delta.mean()) if len(delta) else 0.0
    nb = len(edges) - 1
    table = np.zeros((nb, 2))
    counts = np.zeros((nb, 2), dtype=int)
    b = np.clip(np.searchsorted(edges, conf_tri, side="right") - 1, 0, nb - 1)
    for i in range(nb):
        for e in (0, 1):
            m = (b == i) & (is_empty_pred == bool(e))
            counts[i, e] = int(m.sum())
            table[i, e] = (delta[m].sum() + prior * overall) / (m.sum() + prior)
    return np.maximum(table, 0.0), counts


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg = load_cfg(args)
    v, task = cfg.vision, cfg.task
    dev = vision_device.resolve(v.device)
    feat_dir = f"{cfg.artifacts_dir}/features"
    with open(f"{feat_dir}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    names = meta["class_names"]
    nc = len(names)
    empty_idx = names.index(task.empty_class)

    banner("1. labels")
    ann = cfg.data.annotations_dir
    used = list(cfg.data.train_splits) + [cfg.data.calib_split, cfg.data.eval_split]
    recs = {}
    for s in used:
        by_id = {r.id: r for r in cct20.load_split(ann, s)}
        with open(f"{feat_dir}/{s}_ids.json", encoding="utf-8") as f:
            recs[s] = [by_id[i] for i in json.load(f)]
    y = {s: np.array([names.index(r.label) for r in recs[s]], dtype=np.int64) for s in used}
    for s in used:
        print(f"  {s:<11} {len(y[s]):6d} frames, {np.mean(y[s] == empty_idx) * 100:5.1f}% empty")
    y_train = np.concatenate([y[s] for s in cfg.data.train_splits])
    cal, ev = cfg.data.calib_split, cfg.data.eval_split

    banner("2. exit heads")
    print(f"  device: {vision_device.prepare(dev)}")
    heads, grid_rows, outs = {}, [], {}
    for ri in range(len(v.resolutions)):
        for ti in range(len(v.taps)):
            X_tr = np.concatenate([load_features(feat_dir, s, ri, ti) for s in cfg.data.train_splits])
            print(f"  resolution {meta['input_sizes'][ri]}, exit {ti} (tap {v.taps[ti]}, d={X_tr.shape[1]})")
            th = train_head(X_tr, y_train, nc, v.head_hidden, v.dropout, v.epochs, v.lr,
                            v.weight_decay, v.seed, log=print, device=dev)
            lg_cal = th.logits(load_features(feat_dir, cal, ri, ti))
            ece_before = expected_calibration_error(softmax(lg_cal).max(1),
                                                    softmax(lg_cal).argmax(1) == y[cal])
            th.temperature = fit_temperature(lg_cal, y[cal])
            p_cal = th.probs(load_features(feat_dir, cal, ri, ti))
            p_ev = th.probs(load_features(feat_dir, ev, ri, ti))
            pred_ev, pred_cal = p_ev.argmax(1), p_cal.argmax(1)
            outs[(ri, ti)] = dict(pred_ev=pred_ev, conf_ev=p_ev.max(1),
                                  pred_cal=pred_cal, conf_cal=p_cal.max(1),
                                  correct_cal=pred_cal == y[cal])
            heads[(ri, ti)] = th
            macs = meta["trunk_macs"][str(ri)][str(v.taps[ti])] + head_macs(X_tr.shape[1], v.head_hidden, nc)
            row = dict(resolution=meta["input_sizes"][ri][0], exit=ti, tap=v.taps[ti],
                       mmacs=round(macs / 1e6, 2), temperature=round(th.temperature, 3),
                       acc_calib=float(np.mean(pred_cal == y[cal])),
                       acc_eval=float(np.mean(pred_ev == y[ev])),
                       macro_f1_eval=macro_f1(y[ev], pred_ev),
                       animal_vs_empty_acc_eval=float(np.mean((pred_ev == empty_idx) == (y[ev] == empty_idx))),
                       ece_calib_before=ece_before,
                       ece_calib_after=expected_calibration_error(p_cal.max(1), pred_cal == y[cal]))
            grid_rows.append(row)
            outs[(ri, ti)]["macs"] = macs
            print(f"    eval acc {row['acc_eval']:.3f}  macro-F1 {row['macro_f1_eval']:.3f}  "
                  f"T={row['temperature']}  ECE {row['ece_calib_before']:.3f} -> {row['ece_calib_after']:.3f}")
    write_csv(f"{cfg.results_dir}/tables/accuracy_grid.csv", grid_rows)

    banner("3. operating points and refinement gains")
    ops = {"triage": tuple(v.triage_op), **{k: tuple(x) for k, x in v.refine_ops.items()}}
    tri = outs[ops["triage"]]
    values_cal = frame_values(y[cal], names, task)
    edges = np.unique(np.quantile(tri["conf_cal"], np.linspace(0, 1, v.n_conf_bins + 1)))
    edges[0], edges[-1] = 0.0, 1.0 + 1e-9
    save = dict(class_names=np.array(names), ops=np.array(list(ops)),
                eval_ids=np.array([r.id for r in recs[ev]]), eval_labels=y[ev],
                eval_values=frame_values(y[ev], names, task),
                eval_locations=np.array([r.location for r in recs[ev]]),
                eval_timestamps=np.array([r.timestamp.isoformat() for r in recs[ev]]),
                gain_edges=edges)
    gain_rows = []
    for op, key in ops.items():
        o = outs[key]
        save[f"pred_{op}"] = o["pred_ev"]
        save[f"conf_{op}"] = o["conf_ev"].astype(np.float32)
        save[f"macs_{op}"] = np.int64(o["macs"])
        print(f"  {op:<7} (resolution idx {key[0]}, exit {key[1]}): {o['macs'] / 1e6:7.1f} MMACs, "
              f"eval acc {np.mean(o['pred_ev'] == y[ev]):.3f}")
        if op == "triage":
            continue
        table, counts = gain_table(tri["conf_cal"], tri["correct_cal"], o["correct_cal"],
                                   tri["pred_cal"] == empty_idx, values_cal, edges, v.gain_prior_count)
        save[f"gain_{op}"] = table
        for i in range(table.shape[0]):
            for e in (0, 1):
                gain_rows.append(dict(op=op, conf_lo=round(edges[i], 4), conf_hi=round(edges[i + 1], 4),
                                      triage_said_empty=e, expected_gain=round(table[i, e], 5),
                                      calib_frames=int(counts[i, e])))
    write_csv(f"{cfg.results_dir}/tables/refinement_gain.csv", gain_rows)
    np.savez_compressed(f"{cfg.artifacts_dir}/outcomes.npz", **save)
    # Always stored as CPU tensors, so a checkpoint from a GPU run loads anywhere.
    torch.save({f"{ri}_{ti}": dict(state={k: t.detach().cpu()
                                          for k, t in h.head.state_dict().items()},
                                   mean=h.mean, std=h.std, temperature=h.temperature)
                for (ri, ti), h in heads.items()}, f"{cfg.artifacts_dir}/heads.pt")
    print(f"  wrote {cfg.artifacts_dir}/outcomes.npz and heads.pt")


if __name__ == "__main__":
    main()
