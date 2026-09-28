"""Serengeti metadata parsing and the pre-registered selection rule, on a
synthetic COCO Camera Traps file (no download)."""
from datetime import datetime, timedelta

import numpy as np

from sunsched.config import Config, apply_corpus
from sunsched.data import serengeti as S


def fake_meta(n_cams=12, per_cam=60, seed=0):
    rng = np.random.default_rng(seed)
    cats = [dict(id=0, name="empty"), dict(id=1, name="zebra"), dict(id=2, name="Wildebeest"),
            dict(id=3, name="human")]
    images, anns = [], []
    k = 0
    for c in range(n_cams):
        cam = f"C{c:02d}"
        for s in range(per_cam // 3):
            seq = f"{cam}_seq{s}"
            label = int(rng.choice([0, 0, 0, 1, 2, 3]))
            t0 = datetime(2015, 1, 1) + timedelta(days=int(rng.integers(0, 300)), hours=int(rng.integers(0, 24)))
            for f in range(3):
                iid = f"img{k}"
                k += 1
                images.append(dict(id=iid, file_name=f"S10/{cam}/{cam}_R1/S10_{cam}_R1_IMAG{k:04d}.JPG",
                                   datetime=(t0 + timedelta(seconds=f)).strftime("%Y-%m-%d %H:%M:%S"),
                                   seq_id=seq, frame_num=f + 1))
                anns.append(dict(image_id=iid, category_id=label))
    images.append(dict(id="bad1", file_name="S10/C00/C00_R1/x.JPG", datetime="2015-02-02 01:00:00",
                       corrupt=True))
    images.append(dict(id="bad2", file_name="S10/C00/C00_R1/y.JPG", datetime="", seq_id="z"))
    anns.append(dict(image_id="bad2", category_id=1))
    return dict(images=images, annotations=anns, categories=cats)


def cfg_for_test():
    cfg = apply_corpus(Config(), "serengeti")
    cfg.data.serengeti_train_images = 200
    cfg.data.serengeti_train_empty_frac = 0.2
    cfg.data.serengeti_calib_cameras = 2
    cfg.data.serengeti_eval_budget = 250
    cfg.data.serengeti_min_eval_cameras = 2
    return cfg


def test_parse_records():
    records, names, stats = S.parse_records(fake_meta())
    assert stats["corrupt"] == 1 and stats["no_timestamp"] == 1
    assert names == sorted(names) and "empty" in names and "wildebeest" in names  # lower-cased
    assert all(r.location.startswith("C") for r in records), "location is derived from the path"
    assert len({r.location for r in records}) == 12


def test_selection_is_disjoint_deterministic_and_within_budget():
    records, _, _ = S.parse_records(fake_meta())
    cams = sorted({r.location for r in records})
    splits = {"train": set(cams[:6]), "val": set(cams[6:])}
    cfg = cfg_for_test()
    a = S.select(records, splits, cfg)
    b = S.select(records, splits, cfg)
    assert [r.id for r in a["eval"]] == [r.id for r in b["eval"]], "selection must be deterministic"
    locs = {k: {r.location for r in v} for k, v in a.items()}
    assert not (locs["train"] & locs["eval"]) and not (locs["calib"] & locs["eval"])
    assert not (locs["train"] & locs["calib"])
    assert locs["train"] <= splits["train"] and locs["eval"] <= splits["val"]
    assert len(a["eval"]) <= cfg.data.serengeti_eval_budget
    assert len(a["train"]) <= cfg.data.serengeti_train_images
    # whole sequences only, so near-duplicate burst frames stay together
    seqs = {}
    for r in a["train"]:
        seqs.setdefault(r.seq_id, 0)
        seqs[r.seq_id] += 1
    assert all(v == 3 for v in seqs.values())
    # eval keeps every image of each chosen camera, empties included
    for cam in locs["eval"]:
        assert sum(r.location == cam for r in a["eval"]) == sum(r.location == cam for r in records)


def test_cache_path_is_flat_and_unique():
    cfg = cfg_for_test()
    p1 = S.cache_path(cfg, "S10/B04/B04_R1/S10_B04_R1_IMAG0001.JPG")
    p2 = S.cache_path(cfg, "S10/B05/B05_R1/S10_B04_R1_IMAG0001.JPG")
    assert p1 != p2 and "/" not in p1.split("images_small")[-1].strip("/\\")
