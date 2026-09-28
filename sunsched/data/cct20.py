"""Caltech Camera Traps, CCT20 benchmark subset (Beery et al., ECCV 2018).

Why this dataset: every image carries its real capture time, camera location
and burst id, and its label says what is actually in the frame. So a simulated
deployment can replay a real camera: the event arrives when it really did, and
the classifier is scored on the image that was really taken. (The previous
version paired real timestamps with unrelated CIFAR images, which made the
classifier's confidence uninformative about the event it was supposed to
describe.)
"""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import io
import json
import os
import tarfile

import numpy as np

SPLITS = ("train", "cis_val", "trans_val", "cis_test", "trans_test")
TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                     "%Y:%m:%d %H:%M:%S", "%Y/%m/%d %H:%M:%S")


@dataclass
class Record:
    id: str
    file_name: str
    location: str
    timestamp: datetime
    seq_id: str
    frame_num: int
    label: str


def _parse_ts(raw) -> Optional[datetime]:
    if not raw or str(raw).startswith("0000"):
        return None
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(str(raw).strip(), fmt)
        except ValueError:
            continue
    return None


def _split_path(ann_dir: str, split: str) -> str:
    path = os.path.join(ann_dir, f"{split}_annotations.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing {path}\n  run: python pipeline/00_fetch_datasets.py --only annotations")
    return path


def class_names(ann_dir: str) -> List[str]:
    """Sorted category names, identical across all five splits."""
    names = None
    for split in SPLITS:
        with open(_split_path(ann_dir, split), encoding="utf-8") as f:
            cats = sorted(c["name"] for c in json.load(f)["categories"])
        if names is None:
            names = cats
        elif cats != names:
            raise ValueError(f"category list differs in split {split}")
    return names


def load_split(ann_dir: str, split: str) -> List[Record]:
    with open(_split_path(ann_dir, split), encoding="utf-8") as f:
        d = json.load(f)
    cat = {c["id"]: c["name"] for c in d["categories"]}
    by_image: Dict[str, List[str]] = {}
    for a in d["annotations"]:
        by_image.setdefault(str(a["image_id"]), []).append(cat[a["category_id"]])

    out = []
    for img in d["images"]:
        ts = _parse_ts(img.get("date_captured"))
        labels = by_image.get(str(img["id"]))
        if ts is None or not labels:
            continue
        # A few images carry several boxes; the image label is the most common one.
        label = Counter(labels).most_common(1)[0][0]
        out.append(Record(id=str(img["id"]), file_name=img["file_name"],
                          location=str(img["location"]), timestamp=ts,
                          seq_id=str(img.get("seq_id", "")),
                          frame_num=int(img.get("frame_num", 0) or 0), label=label))
    out.sort(key=lambda r: (r.location, r.timestamp, r.frame_num))
    return out


def check_location_disjoint(splits: Dict[str, Sequence[Record]], train_names, other_names):
    """The trans splits must not share a camera with anything used for training."""
    train_locs = set()
    for s in train_names:
        train_locs |= {r.location for r in splits[s]}
    for s in other_names:
        shared = train_locs & {r.location for r in splits[s]}
        if shared:
            raise ValueError(f"split {s} shares camera locations with training: {sorted(shared)}")


def subset_for_quick(records: Sequence[Record], fraction: float, seed: int,
                     keep_locations: int = 0) -> List[Record]:
    """Smaller data for a smoke test.

    For the evaluation split whole locations are kept (their busiest
    `keep_locations`), because a simulated deployment needs a camera's full
    event stream, not a random sample of its frames.
    """
    if keep_locations:
        counts = Counter(r.location for r in records)
        keep = {loc for loc, _ in counts.most_common(keep_locations)}
        return [r for r in records if r.location in keep]
    rng = np.random.default_rng(seed)
    seqs = sorted({r.seq_id for r in records})
    chosen = set(rng.choice(seqs, size=max(1, int(len(seqs) * fraction)), replace=False))
    # Sample whole bursts so near-duplicate frames stay together.
    return [r for r in records if r.seq_id in chosen]


# -- image access ------------------------------------------------------------
def iter_archive(archive_path: str, wanted: Iterable[str]) -> Iterator[Tuple[str, bytes]]:
    """Stream matching JPEGs out of the .tar.gz without extracting it.

    Reads the archive once, sequentially, and yields (file_name, bytes) for
    every member whose base name is wanted. Nothing is written to disk.
    """
    wanted = set(wanted)
    with tarfile.open(archive_path, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if name in wanted:
                f = tar.extractfile(member)
                if f is not None:
                    yield name, f.read()


def iter_directory(images_dir: str, wanted: Iterable[str]) -> Iterator[Tuple[str, bytes]]:
    wanted = set(wanted)
    for root, _, files in os.walk(images_dir):
        for name in files:
            if name in wanted:
                with open(os.path.join(root, name), "rb") as f:
                    yield name, f.read()


def decode(jpeg: bytes, height: int, width: int):
    """Decode and resize. JPEG draft mode decodes at reduced scale directly,
    which is several times faster than decoding at full size then resizing."""
    from PIL import Image
    img = Image.open(io.BytesIO(jpeg))
    img.draft("RGB", (width, height))
    img = img.convert("RGB")
    return img.resize((width, height), Image.BILINEAR)
