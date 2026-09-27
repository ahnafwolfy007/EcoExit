#!/usr/bin/env python
"""Download everything the pipeline needs. Every link is also listed in DATASETS.md.

    python pipeline/00_fetch_datasets.py                  # everything (~6.6 GB)
    python pipeline/00_fetch_datasets.py --only annotations pvgis weights
    python pipeline/00_fetch_datasets.py --only images    # the big one, resumable

Groups:
    annotations  CCT20 annotation JSONs and official splits      3 MB
    images       CCT20 images, downsized to <=1024 px          6.49 GB
    pvgis        hourly irradiance + air temperature per site    ~2 MB per site
    weights      MobileNetV3-Large ImageNet weights               22 MB
    cct_meta     full Caltech Camera Traps metadata (optional)    9 MB

No account, API key or GPU is needed. The image download resumes where it
stopped if interrupted: just run the same command again.
"""
import os
import sys
import tarfile
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sunsched.cli import base_parser, banner, load_cfg, write_json
from sunsched.config import SITES, pvgis_range

LILA = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps"
FILES = {
    "annotations": dict(url=f"{LILA}/eccv_18_annotations.tar.gz",
                        dest="cct20/eccv_18_annotations.tar.gz", bytes=2997071),
    "images": dict(url=f"{LILA}/eccv_18_all_images_sm.tar.gz",
                   dest="cct20/eccv_18_all_images_sm.tar.gz", bytes=6492615601),
    "cct_meta": dict(url=f"{LILA}/labels/caltech_camera_traps.json.zip",
                     dest="cct_full/caltech_camera_traps.json.zip", bytes=None),
}
DEFAULT_GROUPS = ["annotations", "images", "pvgis", "weights"]


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
            headers = {"User-Agent": "SunSched/2.0"}
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:        # server ignored the range: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                mode = "ab" if have else "wb"
                t0, last = time.time(), 0.0
                with open(part, mode) as f:
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
    print(f"  giving up on {url}; re-run this script to resume from {have / 1e9:.2f} GB")
    return False


def fetch_annotations(cfg) -> bool:
    spec = FILES["annotations"]
    path = os.path.join(cfg.data.root, spec["dest"])
    if not download(spec["url"], path, spec["bytes"]):
        return False
    if not os.path.isdir(cfg.data.annotations_dir):
        with tarfile.open(path, "r:gz") as t:
            t.extractall(os.path.dirname(path))
    n = len([f for f in os.listdir(cfg.data.annotations_dir) if f.endswith(".json")])
    print(f"  annotations: {n} split files in {cfg.data.annotations_dir}")
    return n == 5


def fetch_images(cfg) -> bool:
    spec = FILES["images"]
    print("  6.49 GB. Resumable: if it stops, run the same command again.")
    print("  The pipeline reads images straight out of the archive; do not extract it.")
    return download(spec["url"], os.path.join(cfg.data.root, spec["dest"]), spec["bytes"])


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


def fetch_cct_meta(cfg) -> bool:
    spec = FILES["cct_meta"]
    path = os.path.join(cfg.data.root, spec["dest"])
    if not download(spec["url"], path, spec["bytes"]):
        return False
    with zipfile.ZipFile(path) as z:
        z.extractall(os.path.dirname(path))
    return True


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help="annotations images pvgis weights cct_meta")
    args = ap.parse_args()
    cfg = load_cfg(args)
    groups = args.only or DEFAULT_GROUPS

    runners = dict(annotations=lambda: fetch_annotations(cfg), images=lambda: fetch_images(cfg),
                   pvgis=lambda: fetch_pvgis(cfg), weights=fetch_weights,
                   cct_meta=lambda: fetch_cct_meta(cfg))
    results = {}
    for g in groups:
        if g not in runners:
            raise SystemExit(f"unknown group {g}; choose from {sorted(runners)}")
        banner(f"fetch: {g}")
        results[g] = bool(runners[g]())

    banner("summary")
    for g, ok in results.items():
        print(f"  {g:<12} {'ok' if ok else 'FAILED'}")
    write_json(os.path.join(cfg.data.root, "fetch_manifest.json"),
               dict(results=results, files=FILES))
    if not all(results.values()):
        print("\nSome downloads failed. Re-run to resume; see DATASETS.md for manual links.")
        return 1
    print("\nAll data present. Next: python run_sunsched.py --skip-fetch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
