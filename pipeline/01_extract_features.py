#!/usr/bin/env python
"""Run every image through the frozen trunk once and cache pooled exit features.

    python pipeline/01_extract_features.py --corpus cct20        # 57,864 images, CPU ~30-90 min
    python pipeline/01_extract_features.py --corpus serengeti    # the cached subset, ~15-40 min
    python pipeline/01_extract_features.py --corpus cct20 --quick

CCT20 is streamed straight out of its .tar.gz; Serengeti is read from the
small cached copies made by stage 0. Writes outputs/<corpus>/artifacts/features/,
one float16 matrix per (role, resolution, tap).
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
from sunsched.data.corpus import load_corpus
from sunsched.vision.backbone import TapExtractor, input_size, load_trunk, tap_dims, to_batch
from sunsched.vision.cost import trunk_macs_at_taps

ROLES = ("train", "calib", "eval")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--images-dir", default="",
                    help="CCT20 only: read an already-extracted image folder instead of the archive")
    args = ap.parse_args()
    cfg = load_cfg(args)
    v = cfg.vision
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    banner(f"1. {cfg.data.corpus}: roles (location-disjoint)")
    corpus = load_corpus(cfg, quick=args.quick, images_dir=args.images_dir)
    for role in ROLES:
        recs = corpus.roles[role]
        print(f"  {role:<6} {len(recs):6,} images  {len({r.location for r in recs}):3d} cameras  "
              f"{np.mean([r.label == 'empty' for r in recs]) * 100:5.1f}% empty")
    print(f"  classes ({len(corpus.class_names)}): {', '.join(corpus.class_names)}")

    banner("2. trunk and exit taps")
    trunk = load_trunk(v.backbone)
    ext = TapExtractor(trunk, v.taps).eval()
    sizes = [input_size(h, v.aspect) for h in v.resolutions]
    dims = [tap_dims(ext, h, w) for h, w in sizes]
    macs = {str(ri): {str(t): int(m) for t, m in trunk_macs_at_taps(trunk, v.taps, h, w).items()}
            for ri, (h, w) in enumerate(sizes)}
    for ri, (h, w) in enumerate(sizes):
        print(f"  {h}x{w}: tap dims {dims[ri]}, MMACs {[round(m / 1e6, 1) for m in macs[str(ri)].values()]}")

    feat_dir = f"{cfg.artifacts_dir}/features"
    os.makedirs(feat_dir, exist_ok=True)
    where, buf, found = {}, {}, {}
    for role in ROLES:
        recs = corpus.roles[role]
        for i, r in enumerate(recs):
            where[r.file_name] = (role, i)
        buf[role] = {(ri, ti): np.zeros((len(recs), dims[ri][ti]), dtype=np.float16)
                     for ri in range(len(sizes)) for ti in range(len(v.taps))}
        found[role] = np.zeros(len(recs), dtype=bool)

    banner("3. images")
    h_max, w_max = sizes[-1]
    pending, total, done, bad = [], len(where), 0, 0
    t0 = time.time()

    def flush():
        for ri in range(len(sizes)):
            outs = ext(to_batch([p[2][ri] for p in pending]))
            for ti, o in enumerate(outs):
                o = o.numpy().astype(np.float16)
                for k, (role, row, _) in enumerate(pending):
                    buf[role][(ri, ti)][row] = o[k]
        for role, row, _ in pending:
            found[role][row] = True
        pending.clear()

    for name, jpeg in corpus.iter_images(where.keys()):
        try:
            big = cct20.decode(jpeg, h_max, w_max)
            ims = [big if (h, w) == (h_max, w_max) else big.resize((w, h)) for h, w in sizes]
        except Exception:
            bad += 1
            continue
        role, row = where[name]
        pending.append((role, row, ims))
        if len(pending) >= v.batch_size:
            flush()
        done += 1
        if done % 500 == 0:
            el = time.time() - t0
            print(f"  {done:6,}/{total:,}  {done / el:5.1f} img/s  "
                  f"~{(total - done) / max(done / el, 1e-9) / 60:.1f} min left", flush=True)
    if pending:
        flush()

    banner("4. saving")
    missing = {}
    for role in ROLES:
        keep = found[role]
        missing[role] = int((~keep).sum())
        with open(f"{feat_dir}/{role}_ids.json", "w", encoding="utf-8") as f:
            json.dump([r.id for r, k in zip(corpus.roles[role], keep) if k], f)
        for (ri, ti), arr in buf[role].items():
            np.save(f"{feat_dir}/{role}_r{ri}_t{ti}.npy", arr[keep])
        print(f"  {role:<6} {int(keep.sum()):6,} saved, {missing[role]} missing")
    write_json(f"{feat_dir}/meta.json", dict(
        corpus=cfg.data.corpus, backbone=v.backbone, taps=list(v.taps),
        resolutions=list(v.resolutions), input_sizes=[list(x) for x in sizes], dims=dims,
        trunk_macs=macs, class_names=corpus.class_names, quick=bool(args.quick),
        undecodable=bad, missing=missing, corpus_stats=corpus.stats))
    print(f"  done in {(time.time() - t0) / 60:.1f} min; {bad} undecodable images skipped")
    if sum(missing.values()) > 0.02 * total:
        print("  WARNING: more than 2% of images are missing. For CCT20 the archive may be\n"
              "  incomplete; for Serengeti re-run stage 0, which resumes the image download.")


if __name__ == "__main__":
    main()
