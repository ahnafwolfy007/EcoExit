#!/usr/bin/env python
"""Run the whole SunSched pipeline for one corpus.

    python run_sunsched.py --corpus cct20 --quick       # smoke test on the development corpus
    python run_sunsched.py --corpus cct20               # development run: tune and debug here
    python run_sunsched.py --corpus serengeti           # the pre-registered test: run once, committed code
    python run_sunsched.py --corpus cct20 --from 4      # resume from a stage

Stages:
    0 download data              4 main grid (autonomy x ratio x policy)
    1 extract image features     5 sensitivity, sites, alpha
    2 train heads + detector     6 pre-registered hypotheses (exit 2 = primary claim fails)
    3 weather, streams, H1       7 report -> outputs/<corpus>/results/REPORT.md
"""
import argparse
import subprocess
import sys
import time

STAGES = [
    (0, "download data", "pipeline/00_fetch_datasets.py"),
    (1, "extract image features", "pipeline/01_extract_features.py"),
    (2, "train heads and triage detector", "pipeline/02_train_exits.py"),
    (3, "weather, capture streams, H1", "pipeline/03_build_environment.py"),
    (4, "main grid", "pipeline/04_run_experiments.py"),
    (5, "sensitivity, sites, alpha", "pipeline/05_sweeps.py"),
    (6, "pre-registered hypotheses", "pipeline/06_hypotheses.py"),
    (7, "report", "pipeline/07_make_report.py"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=["cct20", "serengeti"], default="cct20")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-sweeps", action="store_true", help="skip stage 5")
    ap.add_argument("--from", dest="start", type=int, default=0, help="first stage to run")
    ap.add_argument("--solar", choices=["pvgis", "analytic"], default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="./data")
    args = ap.parse_args()

    common = ["--corpus", args.corpus, "--data", args.data]
    if args.out:
        common += ["--out", args.out]
    if args.quick:
        common.append("--quick")
    if args.solar:
        common += ["--solar", args.solar]
    if args.workers:
        common += ["--workers", str(args.workers)]

    t_all, verdict = time.time(), None
    for idx, label, script in STAGES:
        if idx < args.start or (idx == 0 and args.skip_fetch) or (idx == 5 and args.skip_sweeps):
            continue
        cmd = [sys.executable, script] + common
        print(f"\n{'#' * 72}\n# {args.corpus} stage {idx}: {label}\n# {' '.join(cmd)}\n{'#' * 72}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        if idx == 6 and rc in (0, 2):
            verdict = "HOLDS" if rc == 0 else "DOES NOT HOLD"
        elif rc != 0:
            print(f"\nstage {idx} ({label}) failed with exit code {rc}. Fix the error above, then resume:\n"
                  f"  python run_sunsched.py --corpus {args.corpus} --from {idx}"
                  + (" --quick" if args.quick else ""))
            sys.exit(rc)
        print(f"# stage {idx} finished in {(time.time() - t0) / 60:.1f} min", flush=True)

    out = args.out or f"./outputs/{args.corpus}{'_quick' if args.quick else ''}"
    print(f"\nall stages done in {(time.time() - t_all) / 60:.1f} min")
    print(f"report: {out}/results/REPORT.md")
    if verdict:
        role = "TEST -- this decides" if args.corpus == "serengeti" and not args.quick else "development only"
        print(f"primary claim (H1-H3): {verdict}  [{role}]  ({out}/results/tables/hypotheses.json)")


if __name__ == "__main__":
    main()
