#!/usr/bin/env python
"""Run the whole SunSched pipeline in order.

    python run_sunsched.py --quick          # smoke test (after downloading the data)
    python run_sunsched.py                  # full run
    python run_sunsched.py --skip-fetch     # data already downloaded
    python run_sunsched.py --from 4         # resume from a stage

Stages:
    0 download datasets          4 main comparison
    1 extract image features     5 frontiers, regimes, sensitivity
    2 train exit heads           6 kill test (exit 2 = verdict FAIL, not a crash)
    3 build weather and streams  7 report -> outputs/results/REPORT.md
"""
import argparse
import subprocess
import sys
import time

STAGES = [
    (0, "download datasets", "pipeline/00_fetch_datasets.py"),
    (1, "extract image features", "pipeline/01_extract_features.py"),
    (2, "train exit heads", "pipeline/02_train_exits.py"),
    (3, "build weather and capture streams", "pipeline/03_build_environment.py"),
    (4, "main comparison", "pipeline/04_run_experiments.py"),
    (5, "frontiers, regimes, sensitivity", "pipeline/05_sweeps.py"),
    (6, "kill test", "pipeline/06_kill_test.py"),
    (7, "report", "pipeline/07_make_report.py"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-sweeps", action="store_true", help="skip stage 5 (the longest)")
    ap.add_argument("--from", dest="start", type=int, default=0, help="first stage to run")
    ap.add_argument("--solar", choices=["pvgis", "analytic"], default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="./outputs")
    ap.add_argument("--data", default="./data")
    args = ap.parse_args()

    common = ["--out", args.out, "--data", args.data]
    if args.quick:
        common.append("--quick")
    if args.solar:
        common += ["--solar", args.solar]
    if args.workers:
        common += ["--workers", str(args.workers)]

    t_all = time.time()
    verdict = None
    for idx, label, script in STAGES:
        if idx < args.start or (idx == 0 and args.skip_fetch) or (idx == 5 and args.skip_sweeps):
            continue
        cmd = [sys.executable, script] + (["--out", args.out, "--data", args.data] if idx == 0 else common)
        print(f"\n{'#' * 72}\n# stage {idx}: {label}\n# {' '.join(cmd)}\n{'#' * 72}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        if idx == 6 and rc == 2:
            verdict = "FAIL"
        elif rc != 0:
            print(f"\nstage {idx} ({label}) failed with exit code {rc}. Fix the error above, then "
                  f"resume with: python run_sunsched.py --from {idx}" + (" --quick" if args.quick else ""))
            sys.exit(rc)
        elif idx == 6:
            verdict = "PASS"
        print(f"# stage {idx} finished in {(time.time() - t0) / 60:.1f} min", flush=True)

    print(f"\nall stages done in {(time.time() - t_all) / 60:.1f} min")
    print(f"report: {args.out}/results/REPORT.md")
    if verdict:
        print(f"kill test verdict: {verdict}  ({args.out}/results/tables/kill_test.json)")


if __name__ == "__main__":
    main()
