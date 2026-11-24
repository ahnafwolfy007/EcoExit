"""New LILA test corpora (PREREGISTRATION_v3.md), read from LILA's per-image table.

Written before any image of these corpora was downloaded. Only counts from the
metadata had been seen (WP1, sunsched/data/eligibility.py).

Rules, fixed here:
  cameras   eligible = at least 200 frames and 60 days between first and last
            capture inside the 365-day window with most captures, and 98% valid
            times. Roles by eligibility.split_cameras: 25 eval, 5 calib, rest
            train, balanced over activity tertiles in SHA-256 order.
  frames    eval and calib: every frame of the camera's window, the first 5,000
            in time order. No calendar years are stacked.
  labels    empty, human, vehicle as such. Any other label with at least 50
            images among the chosen training images keeps its name, the others
            become "other_animal". Frames whose label says that nobody could
            tell (unknown, unidentifiable, problem ...) are dropped everywhere.
            An image with several labels takes its first animal label.
  train     whole sequences drawn in a seeded random order from the training
            cameras, up to 10,000 images, 20% empty.
"""
import hashlib
import os
import pickle
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd

from sunsched.data import eligibility as E
from sunsched.data.cct20 import Record

CSV_URL = "https://lila.science/public/lila_image_urls_and_labels.csv.zip"
# corpus -> dataset name pattern (lower case)
DATASETS = {
    "lila_swg": r"swg",
    "lila_orinoquia": r"orinoqu",
    "lila_ammonitor": r"ammonitor",
    "lila_nkhotakota": r"nkhotakota",
    "lila_wellington": r"wellington",
    "lila_idaho": r"idaho",
    "lila_nz": r"trail camera images of new zealand",
    "lila_wcs": r"wcs camera",
    "lila_safrica": r"snapshot (?:camdeboo|karoo|kgalagadi|kruger|mountain zebra)",
}
HUMAN = frozenset({"human", "person", "people", "humans", "homo sapiens"})
VEHICLE = frozenset({"vehicle", "car", "truck", "motorcycle", "bicycle", "bike", "atv"})
N_EVAL, N_CALIB, MIN_TRAIN = 25, 5, 50


def csv_path(cfg) -> str:
    return os.path.join(cfg.data.root, "lila_image_urls_and_labels.csv.zip")


def _dir(cfg) -> str:
    return os.path.join(cfg.data.root, cfg.data.corpus)


def table_path(cfg) -> str:
    return os.path.join(_dir(cfg), "table.pkl")


def simple_label(label: str, common: str) -> str:
    """empty / human / vehicle / unknown / the original label."""
    for x in (label, common):
        if x in E.EMPTY:
            return "empty"
        if x in HUMAN:
            return "human"
        if x in VEHICLE:
            return "vehicle"
    return "unknown" if label in E.NON_ANIMAL else label


def build_table(cfg) -> pd.DataFrame:
    """One row per image of the corpus: image, location, t, seq, frame, label, url, alt."""
    if os.path.exists(table_path(cfg)):
        return pd.read_pickle(table_path(cfg))
    z = zipfile.ZipFile(csv_path(cfg))
    inner = [n for n in z.namelist() if n.endswith(".csv")][0]
    use = ["dataset_name", "url_gcp", "url_aws", "image_id", "sequence_id", "location_id", "frame_num",
           "original_label", "common_name", "datetime"]
    parts = []
    for ch in pd.read_csv(z.open(inner), usecols=use, dtype=str, chunksize=1_000_000, keep_default_na=False):
        m = ch.dataset_name.str.lower().str.contains(DATASETS[cfg.data.corpus], regex=True)
        if m.any():
            parts.append(ch[m])
    if not parts:
        raise SystemExit(f"{cfg.data.corpus}: not in the LILA table")
    d = finish_table(pd.concat(parts, ignore_index=True))
    os.makedirs(_dir(cfg), exist_ok=True)
    d.to_pickle(table_path(cfg))
    return d


def finish_table(raw: pd.DataFrame) -> pd.DataFrame:
    lab = raw.original_label.str.lower().str.strip()
    com = raw.common_name.str.lower().str.strip()
    pairs = pd.Series(list(zip(lab, com)))
    cache = {p: simple_label(*p) for p in pairs.unique()}
    label = pairs.map(cache).values
    rank = pd.Series(label).map(lambda x: {"unknown": 3, "empty": 2, "human": 1, "vehicle": 1}.get(x, 0)).values
    d = pd.DataFrame(dict(image=raw.image_id.values, location=raw.location_id.values, t=E.parse_times(raw.datetime).values,
                          seq=raw.sequence_id.values, frame=pd.to_numeric(raw.frame_num, errors="coerce").fillna(0).astype(int).values,
                          label=label, url=raw.url_gcp.values, alt=raw.url_aws.values, rank=rank))
    d = d.sort_values(["rank"], kind="stable").drop_duplicates("image").drop(columns="rank")
    return d.sort_values(["location", "t", "frame", "image"]).reset_index(drop=True)


def roles(table: pd.DataFrame) -> Dict[str, List[str]]:
    d = table.assign(empty=table.label == "empty")
    cams = E.camera_table(d[["location", "t", "empty"]])
    return E.split_cameras(cams[E.eligible(cams)], N_EVAL, N_CALIB)


def window_frames(g: pd.DataFrame, cap: int = 5000) -> pd.DataFrame:
    g = g[g.t.notna()]
    day = g.t.values.astype("datetime64[s]").astype(np.int64) // E.DAY
    lo, hi = E.best_window(day)
    return g[(day >= lo) & (day < hi)].sort_values(["t", "frame", "image"]).head(cap)


def select(table: pd.DataFrame, cfg) -> Tuple[Dict[str, pd.DataFrame], List[str], dict]:
    known = table[table.label != "unknown"]
    r = roles(table)
    out = {}
    for k in ("eval", "calib"):
        out[k] = pd.concat([window_frames(g) for loc, g in known[known.location.isin(r[k])].groupby("location", sort=True)],
                           ignore_index=True)
    pool = known[known.location.isin(r["train"]) & known.t.notna()]
    key = pool.seq.where(pool.seq.astype(str).str.len() > 0, pool.image)
    groups = {k: g.index.values for k, g in pool.groupby(key.values, sort=True)}
    keys = sorted(groups)
    np.random.default_rng(cfg.vision.seed).shuffle(keys)
    n_empty = int(cfg.data.lila_train_images * cfg.data.lila_train_empty_frac)
    n_animal = cfg.data.lila_train_images - n_empty
    is_empty = (pool.label == "empty")
    take, e, a = [], 0, 0
    for k in keys:
        ix = groups[k]
        if bool(is_empty.loc[ix].all()):
            if e + len(ix) <= n_empty:
                take.append(ix)
                e += len(ix)
        elif a + len(ix) <= n_animal:
            take.append(ix)
            a += len(ix)
        if e >= n_empty and a >= n_animal:
            break
    out["train"] = pool.loc[np.concatenate(take)].reset_index(drop=True)
    n = Counter(out["train"].label)
    keep = {x for x, c in n.items() if c >= MIN_TRAIN or x in ("empty", "human", "vehicle")} | {"empty"}
    for k in out:
        out[k] = out[k].assign(label=out[k].label.where(out[k].label.isin(keep), "other_animal"))
    names = sorted(keep | {"other_animal"})
    stats = dict(images=len(table), dropped_unknown=len(table) - len(known), cameras={k: len(v) for k, v in r.items()},
                 frames={k: len(v) for k, v in out.items()}, classes=len(names),
                 empty_eval=float((out["eval"].label == "empty").mean()))
    return out, names, stats


def to_records(d: pd.DataFrame) -> List[Record]:
    return [Record(id=str(x.image), file_name=str(x.url), location=str(x.location), timestamp=x.t.to_pydatetime(),
                   seq_id=str(x.seq), frame_num=int(x.frame), label=str(x.label)) for x in d.itertuples()]


def load(cfg) -> Tuple[Dict[str, List[Record]], List[str], dict]:
    """Selected records per role, cached next to the table."""
    p = os.path.join(_dir(cfg), "selection.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    sel, names, stats = select(build_table(cfg), cfg)
    alt = {x.url: x.alt for k in sel for x in sel[k].itertuples()}
    res = ({k: to_records(v) for k, v in sel.items()}, names, dict(stats, alt=alt))
    with open(p, "wb") as f:
        pickle.dump(res, f)
    return res


# -- images ---------------------------------------------------------------------
def cache_path(cfg, url: str) -> str:
    return os.path.join(_dir(cfg), "images", hashlib.sha1(url.encode()).hexdigest() + ".jpg")


def quote(url: str) -> str:
    return urllib.parse.quote(url, safe=":/%?=&+,;@~!$'()*")


def _fetch_one(urls, dest, h, w, timeout=60) -> str:
    from sunsched.data.cct20 import decode
    last = None
    for attempt in range(4):
        for u in urls:
            if not u:
                continue
            try:
                req = urllib.request.Request(quote(u), headers={"User-Agent": "SunSched/2.1"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = r.read()
                decode(data, h, w).save(dest + ".part", "JPEG", quality=90)
                os.replace(dest + ".part", dest)
                return "ok"
            except Exception as e:
                last = e
        time.sleep(2 * (attempt + 1))
    return f"failed: {last}"


def download_images(cfg, h, w, workers=24) -> Dict[str, int]:
    recs, _, stats = load(cfg)
    os.makedirs(os.path.join(_dir(cfg), "images"), exist_ok=True)
    todo = sorted({r.file_name for v in recs.values() for r in v if not os.path.exists(cache_path(cfg, r.file_name))})
    print(f"  {len(todo):,} images to download")
    counts, bad, t0 = Counter(), [], time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_fetch_one, (u, stats["alt"].get(u)), cache_path(cfg, u), h, w): u for u in todo}
        for i, f in enumerate(as_completed(futs), 1):
            counts["ok" if f.result() == "ok" else "failed"] += 1
            if f.result() != "ok":
                bad.append(f"{futs[f]}\t{f.result()}\n")
            if i % 1000 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i:,}/{len(todo):,}  {rate:.1f} img/s  ~{(len(todo) - i) / max(rate, 1e-9) / 60:.0f} min left", flush=True)
    if bad:
        with open(os.path.join(_dir(cfg), "download_failures.txt"), "w", encoding="utf-8") as fh:
            fh.writelines(bad)
    return dict(counts)


def iter_cached(cfg, wanted: Iterable[str]) -> Iterator[Tuple[str, bytes]]:
    for u in wanted:
        p = cache_path(cfg, u)
        if os.path.exists(p):
            with open(p, "rb") as f:
                yield u, f.read()
