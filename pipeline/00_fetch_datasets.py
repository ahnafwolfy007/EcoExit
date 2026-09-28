#!/usr/bin/env python
"""Download everything the pipeline needs. Every link is also listed in DATASETS.md.

    python pipeline/00_fetch_datasets.py --corpus cct20        # development corpus (~6.6 GB)
    python pipeline/00_fetch_datasets.py --corpus serengeti    # test corpus (~17 GB transferred, ~1 GB kept)
    python pipeline/00_fetch_datasets.py --corpus cct20 --only annotations pvgis

Groups for cct20:     annotations images pvgis weights
Groups for serengeti: metadata images pvgis weights

No account, API key or GPU is needed. Every download resumes if interrupted:
run the same command again.
"""
import os
import sys
import tarfile
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sunsched.cli import banner, base_parser, load_cfg, write_json
from sunsched.config import SITES, pvgis_range

LILA = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps"
CCT_FILES = {
    "annotations": dict(url=f"{LILA}/eccv_18_annotations.tar.gz",
                        dest="cct20/eccv_18_annotations.tar.gz", bytes=2997071),
    "images": dict(url=f"{LILA}/eccv_18_all_images_sm.tar.gz",
                   dest="cct20/eccv_18_all_images_sm.tar.gz", bytes=6492615601),
}


def download(url: str, path: str, expected: int = None) -> bool:
    """HTTP download with resume: an interrupted .part file is continued."""
    import urllib.request

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and (expected is None or os.path.getsize(path) == expected):
        print(f"  already present: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
        return True
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    for attempt in range(1, 6):
        try:
            headers = {"User-Agent": "SunSched/2.1"}
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:        # server ignored the range: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                t0, last = time.time(), 0.0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if now - last > 2 or have == total:
                            rate = (have / 1e6) / max(now - t0, 1e-6)
                            pct = 100.0 * have / total if total else 0.0
                            sys.stdout.write(f"\r  {pct:5.1f}%  {have / 1e9:.2f}/{total / 1e9:.2f} GB  "
                                             f"{rate:.1f} MB/s   ")
                            sys.stdout.flush()
                            last = now
            print()
            if expected is not None and have != expected:
                raise IOError(f"size {have} != expected {expected}")
            os.replace(part, path)
            return True
        except Exception as e:
            print(f"\n  attempt {attempt} failed: {e}")
            have = os.path.getsize(part) if os.path.exists(part) else 0
            time.sleep(min(30, 5 * attempt))
    print(f"  giving up on {url}; re-run this script to resume")
    return False


# -- CCT20 (development) ---------------------------------------------------------
def cct_annotations(cfg) -> bool:
    spec = CCT_FILES["annotations"]
    path = os.path.join(cfg.data.root, spec["dest"])
    if not download(spec["url"], path, spec["bytes"]):
        return False
    if not os.path.isdir(cfg.data.annotations_dir):
        with tarfile.open(path, "r:gz") as t:
            t.extractall(os.path.dirname(path))
    n = len([f for f in os.listdir(cfg.data.annotations_dir) if f.endswith(".json")])
    print(f"  {n} split files in {cfg.data.annotations_dir}")
    return n == 5


def cct_images(cfg) -> bool:
    spec = CCT_FILES["images"]
    print("  6.49 GB, resumable. Do not extract it; the pipeline streams from the archive.")
    return download(spec["url"], os.path.join(cfg.data.root, spec["dest"]), spec["bytes"])


# -- Snapshot Serengeti (test) -----------------------------------------------------
def serengeti_metadata(cfg) -> bool:
    from sunsched.data import serengeti as S
    ok = download(S.metadata_url(cfg.data.serengeti_season), S.metadata_zip_path(cfg))
    ok = download(S.SPLITS_URL, S.splits_path(cfg)) and ok
    if ok:
        records, names, stats = S.parse_records(S.load_metadata(cfg))
        print(f"  season {cfg.data.serengeti_season}: {stats}")
    return ok


def serengeti_images(cfg) -> bool:
    from sunsched.data import serengeti as S
    from sunsched.vision.backbone import input_size
    records, names, _ = S.parse_records(S.load_metadata(cfg))
    roles = S.select(records, S.load_splits(cfg), cfg)
    for role, recs in roles.items():
        print(f"  {role:<6} {len(recs):6,} images from {len({r.location for r in recs}):3d} cameras")
    h, w = input_size(max(cfg.vision.resolutions), cfg.vision.aspect)
    wanted = [r for recs in roles.values() for r in recs]
    counts = S.download_images(wanted, cfg, h, w, workers=8)
    print(f"  {counts}")
    return counts.get("failed", 0) <= 0.01 * max(len(wanted), 1)


# -- shared ------------------------------------------------------------------------
def fetch_pvgis(cfg) -> bool:
    from sunsched.env.solar import fetch_pvgis
    start, end = pvgis_range()
    ok = True
    for site, s in SITES.items():
        try:
            data = fetch_pvgis(s["lat"], s["lon"], start, end, cfg.data.pvgis_dir)
            h = data["outputs"]["hourly"]
            print(f"  {site:<11} {len(h):6d} hourly records, has T2m: {'T2m' in h[0]}")
        except Exception as e:
            ok = False
            print(f"  {site:<11} FAILED: {e}")
    return ok


def fetch_weights() -> bool:
    try:
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        print("  MobileNetV3-Large ImageNet weights cached by torchvision")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args)
    if cfg.data.corpus == "cct20":
        runners = dict(annotations=lambda: cct_annotations(cfg), images=lambda: cct_images(cfg),
                       pvgis=lambda: fetch_pvgis(cfg), weights=fetch_weights)
    else:
        runners = dict(metadata=lambda: serengeti_metadata(cfg), images=lambda: serengeti_images(cfg),
                       pvgis=lambda: fetch_pvgis(cfg), weights=fetch_weights)
    groups = args.only or list(runners)
    results = {}
    for g in groups:
        if g not in runners:
            raise SystemExit(f"unknown group {g} for corpus {cfg.data.corpus}; choose from {sorted(runners)}")
        banner(f"fetch {cfg.data.corpus}: {g}")
        results[g] = bool(runners[g]())

    banner("summary")
    for g, ok in results.items():
        print(f"  {g:<12} {'ok' if ok else 'FAILED'}")
    write_json(os.path.join(cfg.data.root, f"fetch_manifest_{cfg.data.corpus}.json"), dict(results=results))
    if not all(results.values()):
        print("\nSome downloads failed. Re-run to resume; see DATASETS.md for manual links.")
        return 1
    print(f"\nAll data present. Next: python run_sunsched.py --corpus {cfg.data.corpus} --skip-fetch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
