"""Shared command-line plumbing for the scripts under pipeline/."""
import argparse
import csv
import json
import os
from typing import Iterable, List

from sunsched.config import Config, apply_corpus, quick


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--corpus", choices=["cct20", "serengeti"], default="cct20",
                    help="cct20 = development corpus, serengeti = pre-registered test corpus")
    ap.add_argument("--quick", action="store_true",
                    help="smoke-test preset: fewer images, 2 weather years, 90-day season")
    ap.add_argument("--out", default=None,
                    help="where artifacts/ and results/ go (default ./outputs/<corpus>)")
    ap.add_argument("--data", default="./data", help="where the downloaded datasets live")
    ap.add_argument("--solar", choices=["pvgis", "analytic"], default=None,
                    help="irradiance source (default pvgis; analytic is an offline fallback)")
    ap.add_argument("--workers", type=int, default=0, help="parallel workers (0 = cores - 1)")
    return ap


def load_cfg(args) -> Config:
    cfg = Config()
    corpus = getattr(args, "corpus", "cct20")
    apply_corpus(cfg, corpus)
    if getattr(args, "quick", False):
        cfg = quick(cfg)
    cfg.out_dir = args.out or f"./outputs/{corpus}{'_quick' if getattr(args, 'quick', False) else ''}"
    root = args.data.rstrip("/\\")
    cfg.data.root = root
    cfg.data.cct20_dir = f"{root}/cct20"
    cfg.data.images_archive = f"{root}/cct20/eccv_18_all_images_sm.tar.gz"
    cfg.data.annotations_dir = f"{root}/cct20/eccv_18_annotation_files"
    cfg.data.serengeti_dir = f"{root}/serengeti"
    cfg.data.pvgis_dir = f"{root}/pvgis"
    if getattr(args, "solar", None):
        cfg.solar.source = args.solar
    if getattr(args, "workers", 0):
        cfg.experiment.n_workers = args.workers
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    os.makedirs(f"{cfg.results_dir}/tables", exist_ok=True)
    os.makedirs(f"{cfg.results_dir}/figures", exist_ok=True)
    return cfg


def write_csv(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys and k not in ("traceback", "cfg"):
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


STRING_KEYS = {"policy", "location", "site", "solar_source", "error", "tag_param",
               "tag_sweep", "tag_group", "tag_level", "versus", "metric"}


def read_csv(path: str) -> List[dict]:
    """Numbers become floats, except identifier columns: camera location IDs
    look numeric ("46") and must stay strings or pairing across files breaks."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in STRING_KEYS:
                continue
            try:
                r[k] = float(v) if v not in ("", None) else float("nan")
            except (TypeError, ValueError):
                pass
    return rows


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def banner(title: str):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72, flush=True)
