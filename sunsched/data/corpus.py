"""One interface over both corpora: records by role, class names, image bytes.

Roles are always "train", "calib" and "eval", and are location-disjoint. Train
cameras fit the heads, calib cameras fit temperatures and refinement gains,
and eval cameras are the simulated deployments.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Tuple

from sunsched.data import cct20, serengeti
from sunsched.data.cct20 import Record


@dataclass
class Corpus:
    name: str
    class_names: List[str]
    roles: Dict[str, List[Record]]
    all_records: List[Record]            # every labelled capture, for the timing analysis
    iter_images: Callable[[Iterable[str]], Iterator[Tuple[str, bytes]]]
    stats: dict


def _check_disjoint(roles: Dict[str, List[Record]]):
    locs = {k: {r.location for r in v} for k, v in roles.items()}
    for a in locs:
        for b in locs:
            if a < b and locs[a] & locs[b]:
                raise ValueError(f"roles {a} and {b} share cameras: {sorted(locs[a] & locs[b])[:5]}")


def load_corpus(cfg, quick: bool = False, images_dir: str = "") -> Corpus:
    if cfg.data.corpus == "cct20":
        ann = cfg.data.annotations_dir
        splits = {s: cct20.load_split(ann, s) for s in cct20.SPLITS}
        train = [r for s in cfg.data.train_splits for r in splits[s]]
        calib, ev = splits[cfg.data.calib_split], splits[cfg.data.eval_split]
        if quick:
            train = cct20.subset_for_quick(train, 0.2, cfg.vision.seed)
            ev = cct20.subset_for_quick(ev, 1.0, cfg.vision.seed, keep_locations=3)
        roles = dict(train=train, calib=calib, eval=ev)
        src = images_dir or cfg.data.images_dir
        if src:
            it = lambda wanted: cct20.iter_directory(src, wanted)
        else:
            it = lambda wanted: cct20.iter_archive(cfg.data.images_archive, wanted)
        corpus = Corpus("cct20", cct20.class_names(ann), roles,
                        [r for s in cct20.SPLITS for r in splits[s]], it, {})
    elif cfg.data.corpus == "serengeti":
        records, names, stats = serengeti.parse_records(serengeti.load_metadata(cfg))
        roles = serengeti.select(records, serengeti.load_splits(cfg), cfg)
        corpus = Corpus("serengeti", names, roles, records,
                        lambda wanted: serengeti.iter_cached(cfg, wanted), stats)
    else:
        raise ValueError(f"unknown corpus {cfg.data.corpus}")
    _check_disjoint(corpus.roles)
    return corpus
