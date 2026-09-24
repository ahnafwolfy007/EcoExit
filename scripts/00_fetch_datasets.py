#!/usr/bin/env python
"""Download every dataset the pipeline needs.

Run this once before anything else. Every link is listed in DATASETS.md with
its size and purpose; this script just fetches them into `./data` and verifies
what it got.

    python scripts/00_fetch_datasets.py                 # everything except Serengeti
    python scripts/00_fetch_datasets.py --all           # including Serengeti (188 MB)
    python scripts/00_fetch_datasets.py --only cifar    # one group
    python scripts/00_fetch_datasets.py --skip pvgis    # offline machine

Groups: cifar, caltech, serengeti, pvgis

Nothing here needs an account, an API key or a GPU. Total download is about
180 MB without Serengeti, about 370 MB with it.
"""
import argparse
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecoexit.config import Config, SITES


LILA = "https://storage.googleapis.com/public-datasets-lila"

SOURCES = {
    "caltech_metadata": dict(
        url=f"{LILA}/caltechcameratraps/labels/caltech_camera_traps.json.zip",
        dest="camera_traps/caltech_camera_traps.json.zip",
        approx_mb=9.0,
        unzip=True,
        why="Real capture timestamps: 243,100 records, 140 locations, 51.7% empty.",
    ),
    "caltech_splits": dict(
        url=f"{LILA}/caltechcameratraps/CaltechCameraTrapsSplits_v0.json",
        dest="camera_traps/CaltechCameraTrapsSplits_v0.json",
        approx_mb=0.1,
        why="Official location-disjoint splits. Never generate your own.",
    ),
    "serengeti_metadata": dict(
        url=f"{LILA}/snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2_1.json.zip",
        dest="camera_traps/SnapshotSerengeti_S1-11_v2_1.json.zip",
        approx_mb=188.5,
        unzip=True,
        why="Scale and an 11-season temporal axis for the covariate-shift split.",
    ),
    "serengeti_splits": dict(
        url=f"{LILA}/snapshotserengeti-v-2-0/SnapshotSerengetiSplits_v0.json",
        dest="camera_traps/SnapshotSerengetiSplits_v0.json",
        approx_mb=0.1,
        why="Official splits: 179 train / 46 val locations, zero overlap.",
    ),
}

GROUPS = {
    "caltech": ["caltech_metadata", "caltech_splits"],
    "serengeti": ["serengeti_metadata", "serengeti_splits"],
}


def _progress(done: int, total: int, label: str):
    if total <= 0:
        sys.stdout.write(f"\r  {label}: {done / 1e6:.1f} MB")
    else:
        pct = 100.0 * done / total
        sys.stdout.write(f"\r  {label}: {pct:5.1f}%  ({done / 1e6:.1f}/{total / 1e6:.1f} MB)")
    sys.stdout.flush()


def download(url: str, path: str, label: str, force: bool = False) -> bool:
    import urllib.request

    if os.path.exists(path) and not force:
        print(f"  {label}: already present ({os.path.getsize(path) / 1e6:.1f} MB), skipping")
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EcoExit/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    _progress(done, total, label)
        print()
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"\n  {label}: FAILED -- {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def maybe_unzip(path: str) -> str:
    if not path.endswith(".zip"):
        return path
    out_dir = os.path.dirname(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        target = os.path.join(out_dir, names[0])
        if not os.path.exists(target):
            print(f"  unzipping {os.path.basename(path)} ...")
            z.extractall(out_dir)
    return target


def fetch_cifar(cfg: Config) -> bool:
    print("\n[cifar100] 170 MB, via torchvision")
    try:
        from torchvision.datasets import CIFAR100
        CIFAR100(root=cfg.data.root, train=True, download=True)
        CIFAR100(root=cfg.data.root, train=False, download=True)
        print("  ok")
        return True
    except Exception as e:
        print(f"  FAILED -- {e}")
        print("  Manual alternative: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz")
        print(f"  Extract it into {cfg.data.root}/")
        return False


def fetch_camera_traps(cfg: Config, keys, force: bool) -> bool:
    ok = True
    for key in keys:
        spec = SOURCES[key]
        print(f"\n[{key}] ~{spec['approx_mb']} MB -- {spec['why']}")
        path = os.path.join(cfg.data.root, spec["dest"])
        if not download(spec["url"], path, key, force=force):
            ok = False
            continue
        if spec.get("unzip"):
            extracted = maybe_unzip(path)
            verify_metadata(extracted)
    return ok


def verify_metadata(path: str):
    """Parse the file and print the numbers a reviewer would check."""
    if not path.endswith(".json"):
        return
    try:
        from ecoexit.data.camera_traps import load_coco_metadata, build_records
        _, stats = build_records(load_coco_metadata(path))
        print(f"  verified: {stats['n_kept']:,} records kept of {stats['n_total']:,}, "
              f"{stats['n_locations']} locations, "
              f"{stats['empty_fraction'] * 100:.1f}% empty, "
              f"{stats['n_dropped_bad_timestamp']:,} dropped for malformed timestamps")
    except Exception as e:
        print(f"  (could not verify: {e})")


def fetch_pvgis(cfg: Config, force: bool) -> bool:
    print(f"\n[pvgis] real hourly irradiance for {len(SITES)} sites, ~1 MB each, cached")
    from ecoexit.solar.pvgis import fetch_pvgis_hourly

    ok = True
    for site, scfg in SITES.items():
        try:
            series = fetch_pvgis_hourly(scfg["lat"], scfg["lon"], cfg.solar.pvgis_year,
                                        cache_dir=cfg.data.pvgis_cache_dir, force=force)
            print(f"  {site:<10} {len(series):,} hourly records, "
                  f"mean {series.mean():.1f} W/m^2, max {series.max():.0f} W/m^2")
        except Exception as e:
            print(f"  {site:<10} FAILED -- {e}")
            ok = False
    if not ok:
        print("  PVGIS unavailable. The pipeline still runs: leave")
        print("  SolarCfg.source='analytic' to use the offline generator.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include Snapshot Serengeti (188 MB)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: cifar caltech serengeti pvgis")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="groups to skip")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    cfg = Config()
    wanted = set(args.only) if args.only else {"cifar", "caltech", "pvgis"}
    if args.all:
        wanted |= {"cifar", "caltech", "serengeti", "pvgis"}
    wanted -= set(args.skip)

    os.makedirs(cfg.data.root, exist_ok=True)
    os.makedirs(cfg.data.camera_trap_dir, exist_ok=True)

    print("EcoExit dataset fetch")
    print(f"  target directory: {os.path.abspath(cfg.data.root)}")
    print(f"  groups: {', '.join(sorted(wanted))}")

    results = {}
    if "cifar" in wanted:
        results["cifar"] = fetch_cifar(cfg)
    if "caltech" in wanted:
        results["caltech"] = fetch_camera_traps(cfg, GROUPS["caltech"], args.force)
    if "serengeti" in wanted:
        results["serengeti"] = fetch_camera_traps(cfg, GROUPS["serengeti"], args.force)
    if "pvgis" in wanted:
        results["pvgis"] = fetch_pvgis(cfg, args.force)

    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  {k:<12} {'ok' if v else 'FAILED (see above)'}")
    manifest = os.path.join(cfg.data.root, "fetch_manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(dict(results=results, sources=SOURCES), f, indent=2)
    print(f"\nmanifest: {manifest}")

    if all(results.values()):
        print("\nAll good. Next: python run_all.py --quick")
    else:
        print("\nSome downloads failed. The pipeline still runs on what succeeded;")
        print("see DATASETS.md for manual download instructions.")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
