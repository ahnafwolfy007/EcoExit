"""One interface over both corpora for stages 1-3 (paper revision).

Every corpus is presented as named splits of records, using the split names
set by config.apply_corpus: CCT20 keeps its official split names; Serengeti
uses "train", "calib" and "eval", chosen by the pre-registered rule in
sunsched/data/serengeti.py. Train, calibration and evaluation cameras never
overlap (checked here for both corpora).
"""
from typing import Dict, Iterable, Iterator, List, Tuple

from sunsched.data import cct20, serengeti
from sunsched.data.cct20 import Record


def used_splits(cfg) -> List[str]:
    return list(cfg.data.train_splits) + [cfg.data.calib_split, cfg.data.eval_split]


def load_splits(cfg, quick: bool = False) -> Tuple[Dict[str, List[Record]], List[str]]:
    """(records per used split, class names)."""
    if cfg.data.corpus == "cct20":
        ann = cfg.data.annotations_dir
        splits = {s: cct20.load_split(ann, s) for s in cct20.SPLITS}
        names = cct20.class_names(ann)
        cct20.check_location_disjoint(splits, cfg.data.train_splits,
                                      [cfg.data.calib_split, cfg.data.eval_split])
        if quick:
            for s in cfg.data.train_splits:
                splits[s] = cct20.subset_for_quick(splits[s], 0.2, cfg.vision.seed)
            splits[cfg.data.eval_split] = cct20.subset_for_quick(
                splits[cfg.data.eval_split], 1.0, cfg.vision.seed, keep_locations=3)
        return {s: splits[s] for s in used_splits(cfg)}, names
    if cfg.data.corpus == "serengeti":
        records, names, _ = serengeti.parse_records(serengeti.load_metadata(cfg))
        roles = serengeti.select(records, serengeti.load_splits(cfg), cfg)
        locs = {k: {r.location for r in v} for k, v in roles.items()}
        for a in locs:
            for b in locs:
                if a < b and locs[a] & locs[b]:
                    raise ValueError(f"Serengeti roles {a} and {b} share cameras")
        # Only images that were actually downloaded can be used.
        import os
        roles = {k: [r for r in v if os.path.exists(serengeti.cache_path(cfg, r.file_name))]
                 for k, v in roles.items()}
        return roles, names
    raise ValueError(f"unknown corpus {cfg.data.corpus}")


def image_source(cfg, wanted: Iterable[str], images_dir: str = "") -> Iterator[Tuple[str, bytes]]:
    if cfg.data.corpus == "serengeti":
        return serengeti.iter_cached(cfg, wanted)
    src = images_dir or cfg.data.images_dir
    if src:
        return cct20.iter_directory(src, wanted)
    return cct20.iter_archive(cfg.data.images_archive, wanted)


def all_records(cfg) -> List[Record]:
    """Every labelled capture of the corpus, for the day/night timing analysis
    (H1 of PREREGISTRATION.md is computed over every camera of the season)."""
    if cfg.data.corpus == "serengeti":
        records, _, _ = serengeti.parse_records(serengeti.load_metadata(cfg))
        return records
    ann = cfg.data.annotations_dir
    return [r for s in cct20.SPLITS for r in cct20.load_split(ann, s)]
