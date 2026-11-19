"""Snapshot Serengeti: the held-out test corpus of PREREGISTRATION.md.

Swanson et al., "Snapshot Serengeti, high-frequency annotated camera trap
images of 40 mammalian species in an African savanna", Scientific Data 2015,
hosted by LILA BC (https://lila.science/datasets/snapshot-serengeti).

Nothing here was tuned on Serengeti data: the selection rule below was written
and committed before any Serengeti metadata or image was downloaded.

Images are fetched one at a time from LILA's unzipped mirror, downscaled on
arrival and cached small, so only the chosen subset is ever downloaded and the
full-size originals are never stored. The per-image URL format was checked on
2026-09-28 (S1/B04/B04_R1/S1_B04_R1_PICT0012.JPG, 635,647 bytes, both mirrors).
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

from sunsched.data.cct20 import Record, _parse_ts

LILA = "https://storage.googleapis.com/public-datasets-lila"
SPLITS_URL = f"{LILA}/snapshotserengeti-v-2-0/SnapshotSerengetiSplits_v0.json"
IMAGE_BASES = (
    f"{LILA}/snapshotserengeti-unzipped",
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped",
)


def metadata_url(season: str) -> str:
    return f"{LILA}/snapshotserengeti-v-2-0/SnapshotSerengeti{season}.json.zip"


def metadata_zip_path(cfg) -> str:
    return os.path.join(cfg.data.serengeti_dir, f"SnapshotSerengeti{cfg.data.serengeti_season}.json.zip")


def splits_path(cfg) -> str:
    return os.path.join(cfg.data.serengeti_dir, "SnapshotSerengetiSplits_v0.json")


def cache_dir(cfg) -> str:
    return os.path.join(cfg.data.serengeti_dir, "images_small")


def cache_path(cfg, file_name: str) -> str:
    return os.path.join(cache_dir(cfg), file_name.replace("/", "__"))


# -- metadata -------------------------------------------------------------------
def load_metadata(cfg) -> dict:
    path = metadata_zip_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}\n  run: python pipeline/00_fetch_datasets.py --corpus serengeti")
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".json"))
        with z.open(name) as f:
            return json.load(f)


def parse_records(meta: dict) -> Tuple[List[Record], List[str], dict]:
    """COCO Camera Traps JSON -> records, class names, and counts of what was dropped.

    Accepts either `datetime` or `date_captured` for the capture time, and
    derives the camera site from the file path (S10/B04/B04_R1/...) when there
    is no `location` field. Corrupt images, and images without a parseable
    time or any label, are dropped and counted.
    """
    cat = {c["id"]: str(c["name"]).strip().lower() for c in meta.get("categories", [])}
    by_image: Dict[str, List[str]] = {}
    for a in meta.get("annotations", []):
        by_image.setdefault(str(a.get("image_id")), []).append(cat.get(a.get("category_id"), ""))

    records, dropped = [], Counter()
    for img in meta.get("images", []):
        if img.get("corrupt"):
            dropped["corrupt"] += 1
            continue
        ts = _parse_ts(img.get("datetime") or img.get("date_captured"))
        if ts is None:
            dropped["no_timestamp"] += 1
            continue
        labels = [l for l in by_image.get(str(img.get("id")), []) if l]
        if not labels:
            dropped["no_label"] += 1
            continue
        counts = Counter(labels)
        animals = Counter({k: v for k, v in counts.items() if k != "empty"})
        label = animals.most_common(1)[0][0] if animals else "empty"
        fname = str(img["file_name"])
        parts = fname.split("/")
        loc = img.get("location") or (parts[1] if len(parts) > 2 else parts[0])
        records.append(Record(id=str(img["id"]), file_name=fname, location=str(loc), timestamp=ts,
                              seq_id=str(img.get("seq_id", "")),
                              frame_num=int(img.get("frame_num", 0) or 0), label=label))
    records.sort(key=lambda r: (r.location, r.timestamp, r.frame_num))
    names = sorted({r.label for r in records} | {"empty"})
    stats = dict(n_images=len(meta.get("images", [])), n_kept=len(records),
                 empty_fraction=float(np.mean([r.label == "empty" for r in records])) if records else 0.0,
                 n_locations=len({r.location for r in records}), n_classes=len(names), **dropped)
    return records, names, stats


def load_splits(cfg) -> Dict[str, set]:
    with open(splits_path(cfg), encoding="utf-8") as f:
        raw = json.load(f)
    s = raw.get("splits", raw)
    out = {k: {str(x) for x in v} for k, v in s.items()}
    if out.get("train", set()) & out.get("val", set()):
        raise ValueError("official Serengeti splits overlap -- refusing to continue")
    return out


# -- the pre-registered selection rule ----------------------------------------
def select(records: Sequence[Record], splits: Dict[str, set], cfg) -> Dict[str, List[Record]]:
    """Choose training, calibration and evaluation images.

    train  whole sequences sampled from official train locations, up to
           serengeti_train_images images, of which serengeti_train_empty_frac
           are empty (the corpus is ~76% empty; an unbalanced sample would
           teach the heads little besides "empty").
    eval   official val cameras ranked by number of animal images (most first,
           ties by name); cameras are added in that order while the total image
           count stays within serengeti_eval_budget. Every image of a chosen
           camera is kept, empties included, because the simulation replays a
           camera's full capture stream.
    calib  the next serengeti_calib_cameras val cameras in the same ranking.

    Deterministic given the metadata and the config; fetch and extraction both
    call it, so they always agree on the image list.
    """
    d = cfg.data
    train_pool = [r for r in records if r.location in splits["train"]]
    val_pool = [r for r in records if r.location in splits["val"]]

    rng = np.random.default_rng(cfg.vision.seed)
    seqs: Dict[str, List[Record]] = {}
    for r in train_pool:
        seqs.setdefault(r.seq_id or r.id, []).append(r)
    keys = sorted(seqs)
    rng.shuffle(keys)
    n_empty_target = int(d.serengeti_train_images * d.serengeti_train_empty_frac)
    n_animal_target = d.serengeti_train_images - n_empty_target
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
    eval_cams, total = [], 0
    for c in ranking:
        if total + len(per_cam[c]) <= d.serengeti_eval_budget:
            eval_cams.append(c)
            total += len(per_cam[c])
    if len(eval_cams) < d.serengeti_min_eval_cameras:
        raise ValueError(f"only {len(eval_cams)} val cameras fit serengeti_eval_budget="
                         f"{d.serengeti_eval_budget}; raise the budget")
    calib_cams = [c for c in ranking if c not in eval_cams][:d.serengeti_calib_cameras]

    def pick(cams):
        return [r for c in cams for r in per_cam[c]]

    return dict(train=train, calib=pick(calib_cams), eval=pick(eval_cams))


# -- images ---------------------------------------------------------------------
def _fetch_one(file_name: str, dest: str, height: int, width: int, timeout: int = 60) -> str:
    import urllib.request
    from sunsched.data.cct20 import decode

    last = None
    for attempt in range(4):
        for base in IMAGE_BASES:
            try:
                req = urllib.request.Request(f"{base}/{file_name}", headers={"User-Agent": "SunSched/2.1"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = r.read()
                img = decode(data, height, width)
                tmp = dest + ".part"
                img.save(tmp, "JPEG", quality=90)
                os.replace(tmp, dest)
                return "ok"
            except Exception as e:           # try the other mirror, then retry
                last = e
        time.sleep(2 * (attempt + 1))
    return f"failed: {last}"


def download_images(records: Sequence[Record], cfg, height: int, width: int,
                    workers: int = 8) -> Dict[str, int]:
    """Fetch, downscale and cache every image not already cached. Resumable:
    re-running skips what is on disk."""
    os.makedirs(cache_dir(cfg), exist_ok=True)
    todo = [r for r in records if not os.path.exists(cache_path(cfg, r.file_name))]
    print(f"  {len(records) - len(todo):,} already cached, {len(todo):,} to download "
          f"(~{len(todo) * 0.64 / 1024:.1f} GB transferred, ~{len(todo) * 0.03 / 1024:.1f} GB kept)")
    counts = Counter()
    failures = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, r.file_name, cache_path(cfg, r.file_name), height, width): r
                for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            counts["ok" if res == "ok" else "failed"] += 1
            if res != "ok":
                failures.append((futs[fut].file_name, res))
            if i % 250 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i:,}/{len(todo):,}  {rate:.1f} img/s  ~{(len(todo) - i) / max(rate, 1e-9) / 60:.0f} min left",
                      flush=True)
    if failures:
        with open(os.path.join(cfg.data.serengeti_dir, "download_failures.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{n}\t{e}\n" for n, e in failures)
    return dict(counts)


def iter_cached(cfg, wanted: Iterable[str]) -> Iterator[Tuple[str, bytes]]:
    for name in wanted:
        p = cache_path(cfg, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                yield name, f.read()
