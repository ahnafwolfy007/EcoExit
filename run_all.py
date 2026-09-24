#!/usr/bin/env python
"""Run the entire EcoExit pipeline end to end, in order.

    python run_all.py --quick        # ~15 min smoke test, smaller everything
    python run_all.py                # full run
    python run_all.py --real-data    # PVGIS irradiance + camera-trap event timing
    python run_all.py --skip-fetch   # datasets already downloaded

Each stage is a normal script under scripts/ and can be run on its own; this
file just calls them in order with matching flags and stops if one fails.

The gate stage (06) is the one that matters. It decides whether the controller
is actually doing anything, and a non-zero exit there is a real answer, not a
crash -- see RUN_GUIDE.md.
"""
import argparse
import subprocess
import sys
import time


def stages(args):
    days = ["--days", "10"] if args.quick else []
    quick = ["--quick"] if args.quick else []
    real_solar = ["--source", "pvgis"] if args.real_data else []
    real_events = ["--events", "camera_trap"] if args.real_data else []

    out = []
    if not args.skip_fetch:
        out.append(("Download datasets", "scripts/00_fetch_datasets.py", []))
    out += [
        ("Train the elastic backbone", "scripts/01_train_backbone.py", quick),
        ("Profile energy, calibrate confidence, split pools", "scripts/02_profile_energy.py", []),
        ("Build solar traces and conformal forecasts", "scripts/03_build_traces.py",
         days + real_solar + real_events),
        ("PHASE 0 GATE: is the controller doing anything?", "scripts/06_phase0_gate.py",
         quick + (["--profile", args.profile] if args.profile else [])),
        ("Run the full controller comparison", "scripts/04_run_experiments.py",
         ["--skip-rl"] if args.quick else []),
        ("Assemble the report", "scripts/05_make_report.py", []),
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast laptop preset: fewer epochs, shorter traces, fewer seeds")
    ap.add_argument("--real-data", action="store_true",
                    help="use PVGIS irradiance and real camera-trap event timing")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="skip the dataset download stage")
    ap.add_argument("--stop-on-gate-fail", action="store_true",
                    help="halt if the Phase 0 gate fails instead of continuing")
    ap.add_argument("--profile", default=None,
                    help="node hardware class for the gate: sbc, mcu, mcu_small_battery")
    args = ap.parse_args()

    plan = stages(args)
    t_start = time.time()
    gate_failed = False

    for i, (label, script, flags) in enumerate(plan, 1):
        cmd = [sys.executable, script] + flags
        print(f"\n{'=' * 70}\n[{i}/{len(plan)}] {label}\n$ {' '.join(cmd)}\n{'=' * 70}")
        t0 = time.time()
        result = subprocess.run(cmd)

        if "06_phase0_gate" in script and result.returncode == 2:
            # Exit code 2 from the gate means "ran fine, verdict is FAIL".
            gate_failed = True
            print("\n[run_all] GATE FAILED. Everything downstream is descriptive only.")
            if args.stop_on_gate_fail:
                sys.exit(2)
            print("[run_all] continuing so you can inspect the diagnostics.")
        elif result.returncode != 0:
            print(f"\n[run_all] STAGE FAILED: {label} (exit code {result.returncode})")
            sys.exit(result.returncode)

        print(f"[run_all] stage finished in {time.time() - t0:.1f}s")

    print(f"\n[run_all] ALL STAGES DONE in {(time.time() - t_start) / 60:.1f} min.")
    print("[run_all] Open results/REPORT.html for the full report.")
    print("[run_all] Gate verdict: results/tables/gate_verdict.json")
    if gate_failed:
        print("\n[run_all] Reminder: the gate did NOT pass. Read its diagnostics above")
        print("[run_all] before treating any downstream number as a result.")
        sys.exit(2)


if __name__ == "__main__":
    main()
