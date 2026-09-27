#!/usr/bin/env python
"""Stream CCT20 images through the frozen trunk and cache pooled exit features.

    python pipeline/01_extract_features.py            # all 57,864 images (CPU: ~30-60 min)
    python pipeline/01_extract_features.py --quick    # a subset (~5-10 min)

Reads the .tar.gz sequentially without extracting it. Writes
outputs/artifacts/features/, one float16 matrix per (split, resolution, tap).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sunsched.cli import banner, base_parser, load_cfg, write_json
from sunsched.data import cct20
from sunsched.vision.backbone import TapExtractor, input_size, load_trunk, tap_dims, to_batch
from sunsched.vision.cost import trunk_macs_at_taps


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--images-dir", default="",
                    help="read an already-extracted image folder instead of the archive")
    args = ap.parse_args()
    cfg = load_cfg(args)
    v = cfg.vision
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    banner("1. splits")
    ann = cfg.data.annotations_dir
    splits = {s: cct20.load_split(ann, s) for s in cct20.SPLITS}
    names = cct20.class_names(ann)
    cct20.check_location_disjoint(splits, cfg.data.train_splits,
                                  [cfg.data.calib_split, cfg.data.eval_split])
    used = list(cfg.data.train_splits) + [cfg.data.calib_split, cfg.data.eval_split]
    if args.quick:
        for s in cfg.data.train_splits:
            splits[s] = cct20.subset_for_quick(splits[s], 0.2, v.seed)
        splits[cfg.data.eval_split] = cct20.subset_for_quick(splits[cfg.data.eval_split], 1.0, v.seed,
                                                             keep_locations=3)
    for s in used:
        locs = len({r.location for r in splits[s]})
        print(f"  {s:<11} {len(splits[s]):6d} images  {locs:2d} locations")
    print(f"  classes ({len(names)}): {', '.join(names)}")
    print("  trans splits share no camera with training: checked")

    banner("2. trunk and exit taps")
    trunk = load_trunk(v.backbone)
    ext = TapExtractor(trunk, v.taps).eval()
    sizes = [input_size(h, v.aspect) for h in v.resolutions]
    dims = [tap_dims(ext, h, w) for h, w in sizes]
    macs = {str(ri): {str(t): int(m) for t, m in trunk_macs_at_taps(trunk, v.taps, h, w).items()}
            for ri, (h, w) in enumerate(sizes)}
    for ri, (h, w) in enumerate(sizes):
        print(f"  resolution {h}x{w}: tap dims {dims[ri]}, "
              f"MMACs at taps {[round(m / 1e6, 1) for m in macs[str(ri)].values()]}")

    feat_dir = f"{cfg.artifacts_dir}/features"
    os.makedirs(feat_dir, exist_ok=True)
    where = {}                                    # file_name -> (split, row)
    buf = {}
    for s in used:
        for i, r in enumerate(splits[s]):
            where[r.file_name] = (s, i)
        buf[s] = {(ri, ti): np.zeros((len(splits[s]), dims[ri][ti]), dtype=np.float16)
                  for ri in range(len(sizes)) for ti in range(len(v.taps))}
    found = {s: np.zeros(len(splits[s]), dtype=bool) for s in used}

    banner("3. streaming images")
    images_dir = args.images_dir or cfg.data.images_dir
    if images_dir:
        source = cct20.iter_directory(images_dir, where.keys())
        print(f"  reading folder {images_dir}")
    else:
        if not os.path.exists(cfg.data.images_archive):
            raise SystemExit(f"missing {cfg.data.images_archive}\n"
                             f"  run: python pipeline/00_fetch_datasets.py --only images")
        source = cct20.iter_archive(cfg.data.images_archive, where.keys())
        print(f"  streaming {cfg.data.images_archive} (one sequential pass)")

    h_max, w_max = sizes[-1]
    pending = []                                  # (split, row, [PIL per resolution])
    total, done, bad = len(where), 0, 0
    t0 = time.time()

    def flush():
        for ri in range(len(sizes)):
            batch = to_batch([p[2][ri] for p in pending])
            outs = ext(batch)
            for ti, o in enumerate(outs):
                o = o.numpy().astype(np.float16)
                for k, (s, row, _) in enumerate(pending):
                    buf[s][(ri, ti)][row] = o[k]
        for s, row, _ in pending:
            found[s][row] = True
        pending.clear()

    for name, jpeg in source:
        try:
            big = cct20.decode(jpeg, h_max, w_max)
            ims = [big.resize((w, h)) if (h, w) != (h_max, w_max) else big for h, w in sizes]
        except Exception:
            bad += 1
            continue
        s, row = where[name]
        pending.append((s, row, ims))
        if len(pending) >= v.batch_size:
            flush()
        done += 1
        if done % 500 == 0:
            el = time.time() - t0
            print(f"  {done:6d}/{total}  {done / el:5.1f} img/s  ~{(total - done) / max(done / el, 1e-9) / 60:.1f} min left",
                  flush=True)
    if pending:
        flush()

    banner("4. saving")
    missing = {s: int((~found[s]).sum()) for s in used}
    for s in used:
        keep = found[s]
        ids = [r.id for r, k in zip(splits[s], keep) if k]
        with open(f"{feat_dir}/{s}_ids.json", "w", encoding="utf-8") as f:
            json.dump(ids, f)
        for (ri, ti), arr in buf[s].items():
            np.save(f"{feat_dir}/{s}_r{ri}_t{ti}.npy", arr[keep])
        print(f"  {s:<11} {int(keep.sum()):6d} saved, {missing[s]} not found in the archive")
    write_json(f"{feat_dir}/meta.json", dict(
        backbone=v.backbone, taps=list(v.taps), resolutions=list(v.resolutions),
        input_sizes=[list(x) for x in sizes], dims=dims, trunk_macs=macs, class_names=names,
        quick=bool(args.quick), undecodable=bad, missing=missing))
    print(f"  done in {(time.time() - t0) / 60:.1f} min; {bad} undecodable images skipped")
    if sum(missing.values()) > 0.02 * total:
        print("  WARNING: more than 2% of images were not found -- is the archive complete?")


if __name__ == "__main__":
    main()
