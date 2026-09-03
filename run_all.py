#!/usr/bin/env python
"""Run the entire EcoExit pipeline end to end, in order.

    python run_all.py            # full run (~30-90 min on a laptop CPU)
    python run_all.py --quick    # ~10-15 min smoke test, smaller everything

Each stage is a normal script under scripts/ and can also be run on its own
-- this file just calls them in the right order with matching flags and
stops immediately if one of them fails.
"""
import argparse
import subprocess
import sys
import time


STAGES = [
    ("Train the elastic backbone", "scripts/01_train_backbone.py", ["--quick"]),
    ("Profile energy + calibrate confidence", "scripts/02_profile_energy.py", []),
    ("Build solar traces + conformal forecasts", "scripts/03_build_traces.py", ["--days"]),
    ("Run the full controller comparison", "scripts/04_run_experiments.py", ["--skip-rl"]),
    ("Assemble the report", "scripts/05_make_report.py", []),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                     help="fast laptop preset: fewer epochs, shorter traces, skip the RL baselines")
    args = ap.parse_args()

    t_start = time.time()
    for i, (label, script, quick_flags) in enumerate(STAGES, 1):
        cmd = [sys.executable, script]
        if args.quick:
            for flag in quick_flags:
                if flag == "--days":
                    cmd += ["--days", "10"]
                else:
                    cmd.append(flag)
        print(f"\n{'='*70}\n[{i}/{len(STAGES)}] {label}\n$ {' '.join(cmd)}\n{'='*70}")
        t0 = time.time()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[run_all] STAGE FAILED: {label} (exit code {result.returncode})")
            sys.exit(result.returncode)
        print(f"[run_all] stage finished in {time.time()-t0:.1f}s")

    print(f"\n[run_all] ALL STAGES DONE in {(time.time()-t_start)/60:.1f} min.")
    print("[run_all] Open results/REPORT.html to see everything.")


if __name__ == "__main__":
    main()
