"""Slow smoke test: stages 4-7 end to end on a synthetic world (a few minutes).

    python -m sunsched.tests --slow

Builds fake classifier outcomes, capture streams and analytic weather in a
temporary folder, then runs the real pipeline scripts against it. It checks
the plumbing -- the parallel runner, config round-tripping, the autonomy grid,
steady-state passes, the hypothesis tests and the report -- not any result.
"""
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np

from sunsched.cli import read_csv
from sunsched.config import SITES, Config, quick
from sunsched.env.events import slot_in_year
from sunsched.env.solar import analytic_year, save_year

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_fake_artifacts(out_dir: str, n_locations: int = 3, per_location: int = 250, seed: int = 0):
    rng = np.random.default_rng(seed)
    cfg = quick(Config())
    cfg.out_dir = out_dir
    art = cfg.artifacts_dir
    os.makedirs(f"{art}/solar", exist_ok=True)
    os.makedirs(f"{cfg.results_dir}/tables", exist_ok=True)
    names = ["bird", "coyote", "empty"]
    n = n_locations * per_location
    labels = rng.integers(0, 3, n)

    def preds(acc):
        p = labels.copy()
        wrong = rng.random(n) > acc
        p[wrong] = rng.integers(0, 3, wrong.sum())
        return p

    ts = []
    for _ in range(n):
        day = int(rng.integers(100, 180))
        hour = int(rng.choice([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 18, 19, 12]))
        ts.append(datetime(2013, 1, 1) + timedelta(days=day, hours=hour, minutes=int(rng.integers(0, 60))))
    locs = np.repeat([str(40 + i) for i in range(n_locations)], per_location)
    ops = ["triage", "lite", "full"]
    save = dict(class_names=np.array(names), ops=np.array(ops), eval_ids=np.array([str(i) for i in range(n)]),
                eval_labels=labels, eval_values=np.where(labels == 2, 0.1, 1.0), eval_locations=locs,
                eval_timestamps=np.array([t.isoformat() for t in ts]),
                p_animal_triage=np.clip(np.where(labels == 2, 0.2, 0.8) + rng.normal(0, 0.15, n), 0, 1),
                gain_edges=np.array([0.0, 0.33, 0.66, 1.0 + 1e-9]),
                gain_lite=np.array([0.0, 0.05, 0.15]), gain_full=np.array([0.0, 0.08, 0.25]))
    for op, acc, macs in zip(ops, (0.5, 0.7, 0.85), (40e6, 300e6, 600e6)):
        save[f"pred_{op}"] = preds(acc)
        save[f"conf_{op}"] = rng.random(n).astype(np.float32)
        save[f"macs_{op}"] = np.int64(macs)
    np.savez_compressed(f"{art}/outcomes.npz", **save)
    np.savez_compressed(f"{art}/events.npz", eval_slot_in_year=slot_in_year(ts, cfg.solar.slot_minutes),
                        eval_locations=locs)
    for y in cfg.solar.years:
        for site in SITES:
            save_year(analytic_year(site, SITES[site], y, cfg.solar.slot_minutes), f"{art}/solar/{site}_{y}.npz")
    with open(f"{cfg.results_dir}/tables/environment_summary.json", "w", encoding="utf-8") as f:
        json.dump(dict(corpus="cct20", site="cct_region", n_animal_captures=int((labels != 2).sum()),
                       n_cameras=n_locations, events_sun_below_horizon=0.55,
                       events_sun_below_horizon_ci=[0.5, 0.6], events_sun_below_15deg=0.6,
                       events_sun_above_40deg=0.18, energy_sun_below_15deg=0.05,
                       energy_sun_above_40deg=0.55, pearson_hourly_events_vs_energy=-0.2,
                       solar_source="analytic"), f)
    return cfg


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="sunsched_smoke_")
    build_fake_artifacts(tmp)
    common = ["--quick", "--out", tmp, "--solar", "analytic", "--workers", "2"]
    steps = [("04_run_experiments.py", []), ("05_sweeps.py", ["--only", "alpha", "sites"]),
             ("06_hypotheses.py", []), ("07_make_report.py", [])]
    for script, extra in steps:
        print(f"\n--- smoke: {script}", flush=True)
        rc = subprocess.run([sys.executable, os.path.join("pipeline", script)] + common + extra,
                            cwd=ROOT).returncode
        ok = rc in (0, 2) if script == "06_hypotheses.py" else rc == 0
        if not ok:
            print(f"SMOKE FAIL: {script} exited {rc}")
            return 1
    tables = f"{tmp}/results/tables"
    runs = read_csv(f"{tables}/runs.csv")
    errors = [r for r in runs if isinstance(r.get("error"), str) and r["error"]]
    for f in ("summary.csv", "run_manifest.json", "alpha.csv", "sites.csv",
              "hypotheses.json", "hypotheses_cells.csv"):
        if not os.path.exists(f"{tables}/{f}"):
            print(f"SMOKE FAIL: missing {f}")
            return 1
    autonomies = {r["autonomy"] for r in runs}
    if errors or len(autonomies) < 3 or not os.path.exists(f"{tmp}/results/REPORT.md"):
        print(f"SMOKE FAIL: {len(errors)} failed jobs; autonomy levels {sorted(autonomies)}")
        return 1
    print(f"\nSMOKE PASS: {len(runs)} simulated deployments across autonomy {sorted(autonomies)}; "
          f"report at {tmp}/results/REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
