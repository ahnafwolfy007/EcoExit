"""Held-out test corpus A of PREREGISTRATION_v2.md: Caltech Camera Traps
cameras that CCT20 does not use.

Caltech Camera Traps (Beery, Van Horn, Perona, ECCV 2018) has 140 camera
locations in the American Southwest; CCT20, the development corpus of this
project, uses 20 of them. The other 120 have never been loaded, inspected or
scored by this project. This module was written and committed before any of
their metadata or images were downloaded.

Selection rule (fixed here, before download):
  eval   official val locations, minus the 20 CCT20 locations, ranked by number
         of animal images (most first, ties by name), added in that order while
         the total image count stays within cct_new_eval_budget; the budget
         rises in 5,000-image steps until at least cct_new_min_eval_cameras fit.
         Every image of a chosen camera is kept (the simulator replays the
         full capture stream).
  calib  the next cct_new_calib_cameras val locations in the same ranking.
  train  whole sequences sampled from official train locations (any, CCT20
         training cameras included), up to cct_new_train_images images, of
         which cct_new_train_empty_frac are empty.
Images are fetched one at a time from LILA's unzipped mirror and downscaled on
arrival, so the 105 GB archive is never needed.
"""
import io
import json
import os
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from sunsched.data.cct20 import Record
from sunsched.data.serengeti import parse_records  # same COCO Camera Traps parser

LILA = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps"
METADATA_URL = f"{LILA}/labels/caltech_camera_traps.json.zip"
SPLITS_URL = f"{LILA}/CaltechCameraTrapsSplits_v0.json"
IMAGE_BASES = (
    "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images",
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images",
)
# The 20 locations of CCT20 (development corpus), read from its annotation files.
CCT20_LOCATIONS = frozenset({"0", "7", "28", "33", "38", "40", "43", "46", "51", "61", "78", "88",
                             "90", "100", "105", "108", "115", "120", "125", "130"})


def root(cfg) -> str:
    return cfg.data.cct_new_dir


def metadata_zip_path(cfg) -> str:
    return os.path.join(root(cfg), "caltech_camera_traps.json.zip")


def splits_path(cfg) -> str:
    return os.path.join(root(cfg), "CaltechCameraTrapsSplits_v0.json")


def cache_dir(cfg) -> str:
    return os.path.join(root(cfg), "images_small")


def cache_path(cfg, file_name: str) -> str:
    return os.path.join(cache_dir(cfg), file_name.replace("/", "__"))


def load_metadata(cfg) -> dict:
    with zipfile.ZipFile(metadata_zip_path(cfg)) as z:
        name = next(n for n in z.namelist() if n.endswith(".json"))
        with z.open(name) as f:
            return json.load(f)


def load_splits(cfg) -> Dict[str, set]:
    with open(splits_path(cfg), encoding="utf-8") as f:
        raw = json.load(f)
    s = raw.get("splits", raw)
    train, val = set(), set()
    for k, v in s.items():
        if "train" in k.lower():
            train |= {str(x) for x in v}
        elif "val" in k.lower() or "test" in k.lower():
            val |= {str(x) for x in v}
    if train & val:
        raise ValueError("official CCT splits overlap -- refusing to continue")
    return {"train": train, "val": val}


def select(records: Sequence[Record], splits: Dict[str, set], cfg) -> Dict[str, List[Record]]:
    d = cfg.data
    train_pool = [r for r in records if r.location in splits["train"]]
    val_pool = [r for r in records if r.location in splits["val"] and r.location not in CCT20_LOCATIONS]

    rng = np.random.default_rng(cfg.vision.seed)
    seqs: Dict[str, List[Record]] = {}
    for r in train_pool:
        seqs.setdefault(r.seq_id or r.id, []).append(r)
    keys = sorted(seqs)
    rng.shuffle(keys)
    n_empty_target = int(d.cct_new_train_images * d.cct_new_train_empty_frac)
    n_animal_target = d.cct_new_train_images - n_empty_target
    train, n_e, n_a = [], 0, 0
    for k in keys:
        grp = seqs[k]
        is_empty = all(r.label == "empty" for r in grp)
        if is_empty and n_e + len(grp) <= n_empty_target:
            train += grp
            n_e += len(grp)
        elif not is_empty and n_a + len(grp) <= n_animal_target:
            train += grp
            n_a += len(grp)
        if n_e >= n_empty_target and n_a >= n_animal_target:
            break

    per_cam: Dict[str, List[Record]] = {}
    for r in val_pool:
        per_cam.setdefault(r.location, []).append(r)
    ranking = sorted(per_cam, key=lambda c: (-sum(r.label != "empty" for r in per_cam[c]), c))

    def fill(budget):
        cams, total = [], 0
        for c in ranking:
            if total + len(per_cam[c]) <= budget:
                cams.append(c)
                total += len(per_cam[c])
        return cams

    budget = d.cct_new_eval_budget
    eval_cams = fill(budget)
    while len(eval_cams) < d.cct_new_min_eval_cameras:
        budget += 5000
        if budget > 500000:
            raise ValueError("no budget up to 500,000 images admits the minimum number of cameras")
        eval_cams = fill(budget)
    if budget != d.cct_new_eval_budget:
        print(f"  [rule] cct_new_eval_budget raised {d.cct_new_eval_budget:,} -> {budget:,} "
              f"so that {len(eval_cams)} >= {d.cct_new_min_eval_cameras} eval cameras fit")
    calib_cams = [c for c in ranking if c not in eval_cams][:d.cct_new_calib_cameras]

    def pick(cams):
        return [r for c in cams for r in per_cam[c]]

    return dict(train=train, calib=pick(calib_cams), eval=pick(eval_cams))


def _fetch_one(file_name: str, dest: str, height: int, width: int, timeout: int = 60) -> str:
    import urllib.request
    from sunsched.data.cct20 import decode
    last = None
    for attempt in range(4):
        for base in IMAGE_BASES:
            try:
                req = urllib.request.Request(f"{base}/{file_name}", headers={"User-Agent": "SunSched/3.0"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = r.read()
                img = decode(data, height, width)
                tmp = dest + ".part"
                img.save(tmp, "JPEG", quality=90)
                os.replace(tmp, dest)
                return "ok"
            except Exception as e:
                last = e
        time.sleep(2 * (attempt + 1))
    return f"failed: {last}"


def download_images(records: Sequence[Record], cfg, height: int, width: int, workers: int = 16) -> Dict[str, int]:
    os.makedirs(cache_dir(cfg), exist_ok=True)
    todo = [r for r in records if not os.path.exists(cache_path(cfg, r.file_name))]
    print(f"  {len(records) - len(todo):,} cached, {len(todo):,} to download")
    counts, t0 = Counter(), time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, r.file_name, cache_path(cfg, r.file_name), height, width): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            counts["ok" if fut.result() == "ok" else "failed"] += 1
            if i % 1000 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i:,}/{len(todo):,}  {rate:.1f} img/s", flush=True)
    return dict(counts)


def iter_cached(cfg, wanted: Iterable[str]) -> Iterator[Tuple[str, bytes]]:
    for name in wanted:
        p = cache_path(cfg, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                yield name, f.read()
