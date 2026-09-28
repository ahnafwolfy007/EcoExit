# Running SunSched

This guide takes a fresh machine to finished results. Everything runs on a CPU; no
GPU and no hardware are needed.

> The folders `ecoexit/`, `scripts/`, `tests/` and `run_all.py` are the **old v1
> code**. Ignore them. Everything below uses `sunsched/`, `pipeline/` and
> `run_sunsched.py`.

## The protocol, in one paragraph

There are two datasets with different jobs. **CCT20** is the *development* corpus:
run on it, look at the results, and change settings if needed. **Snapshot
Serengeti** is the *test* corpus: it decides the hypotheses in
[PREREGISTRATION.md](PREREGISTRATION.md). It must be run **once**, with committed,
unmodified code, and only after development is finished. If code or settings
change after the Serengeti results have been seen, that second run is
exploratory and has to be reported as such.

## What you need

| | |
|---|---|
| Python | 3.10, 3.11 or 3.12 (3.13+ usually works; use 3.11 if installs fail) |
| Disk | about **12 GB** free (CCT20 archive 6.5 GB, Serengeti cache ~1 GB, features and outputs) |
| Network | about **24 GB** transferred in total (CCT20 6.6 GB, Serengeti ~17 GB), resumable |
| RAM | 8 GB or more |
| CPU | any; more cores make the simulation stages faster |

## 1. Set up

```bash
cd EcoExit-Model
python -m venv venv
```

Activate it:

- Windows PowerShell: `.\venv\Scripts\Activate.ps1`
- Windows cmd: `venv\Scripts\activate.bat`
- Mac/Linux: `source venv/bin/activate`

```bash
pip install -r requirements.txt
```

If installing torch fails, install the CPU build explicitly and run the line above again:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 2. Check the install

```bash
python -m sunsched.tests
```

All **26** tests must pass (seconds, synthetic data). Optionally add a few-minute
end-to-end check of stages 4–7 on a synthetic world:

```bash
python -m sunsched.tests --slow
```

If anything fails, stop and send the output.

## 3. Development: CCT20

```bash
python pipeline/00_fetch_datasets.py --corpus cct20
python run_sunsched.py --corpus cct20 --skip-fetch --quick     # pipeline check, ~20-40 min
python run_sunsched.py --corpus cct20 --skip-fetch             # full development run
```

Outputs go to `outputs/cct20/results/` (and `outputs/cct20_quick/` for the quick
run). This is the place to debug and tune.

**Freeze before testing.** If anything in `sunsched/` or `pipeline/` was changed
during development, commit it now. Stage 4 records the git commit in
`run_manifest.json` and warns if there are uncommitted changes. That record is
what shows the test run used a frozen setup.

## 4. Test: Snapshot Serengeti, run once

```bash
python pipeline/00_fetch_datasets.py --corpus serengeti
python run_sunsched.py --corpus serengeti --skip-fetch
```

Stage 0 downloads the season metadata and official splits. It then applies the
fixed selection rule and fetches only the chosen images (~28,000, about 17 GB
transferred, about 1 GB kept after downscaling). It resumes if interrupted: run
the same command again.

Outputs go to `outputs/serengeti/results/`. **Do not change settings and re-run
after seeing these results**, or the test no longer counts.

Optional pipeline check before the real test, with a smaller image selection:

```bash
python pipeline/00_fetch_datasets.py --corpus serengeti --quick
python run_sunsched.py --corpus serengeti --skip-fetch --quick
```

The quick run looks only at a reduced grid and is labelled "not results"
everywhere. It does not use up the test, but its numbers must not guide any
further changes.

## Rough timing on an 8-core laptop CPU

Your machine will differ.

| Stage | What | CCT20 | Serengeti |
|---|---|---|---|
| 0 | Download | depends on connection | depends on connection (~17 GB) |
| 1 | Images through the frozen network | 30–90 min | 15–40 min |
| 2 | Train heads and detector | 5–15 min | 5–15 min |
| 3 | Weather, capture streams, H1 | 1–5 min | 1–5 min |
| 4 | Main grid: 7 policies × 40 cells × cameras × 5 years × 2 passes | 1–3 h | 1–3 h |
| 5 | Sensitivity, sites, alpha | 3–6 h | 3–6 h |
| 6, 7 | Hypotheses, report | seconds | seconds |

`--skip-sweeps` skips stage 5 for a first pass. The hypotheses and report still
run, but section 5 of the report stays empty until stage 5 has run.

Resume from any stage after a failure:

```bash
python run_sunsched.py --corpus serengeti --skip-fetch --from 4
```

Every stage also runs on its own, e.g. `python pipeline/04_run_experiments.py --corpus serengeti`.
All scripts take `--corpus`, `--quick`, `--workers N`, `--out`, `--data`.

## Results

In `outputs/<corpus>/results/`:

| File | What it is |
|---|---|
| `REPORT.md` | Everything, one page. **Start here.** |
| `tables/hypotheses.json` | The pre-registered verdicts H1–H5 and the claim they allow |
| `tables/hypotheses_cells.csv` | Every cell behind H2–H5 |
| `tables/summary.csv` | Mean and 95% CI of every metric, per autonomy × ratio × policy |
| `tables/runs.csv` | One row per simulated deployment |
| `tables/sensitivity.csv`, `sites.csv`, `alpha.csv` | Stage 5 sweeps |
| `tables/environment_summary.json` | The H1 measurement, with a CI over cameras |
| `tables/run_manifest.json` | Git commit and full configuration of the run |
| `figures/regime_map.png` | Where deferral pays and where a charge ceiling pays |

`outputs/*/artifacts/` holds cached features and model outputs. They are large and
regenerable; do not commit them.

**What to send back:** zip `outputs/cct20/results/` and `outputs/serengeti/results/`
(the `results` folders only, not `artifacts`) and the terminal output.

## Reading the verdict

`run_sunsched.py` ends with `primary claim (H1-H3): HOLDS` or `DOES NOT HOLD`, and
whether that run is the test or only development. Stage 6 exits with code 2
when the primary claim fails. That is a result, not a crash. REPORT.md section 1
states which claim the outcome allows.

## Troubleshooting

**`No module named sunsched`**: run the commands from the `EcoExit-Model` folder
itself.

**PVGIS unreachable**: retry later. If it stays down, `--solar analytic` runs on
synthetic weather, and the report is marked synthetic.

**Serengeti downloads fail for some images**: re-run stage 0; it retries only the
missing ones. `data/serengeti/download_failures.txt` lists what failed. Up to 1%
failures are tolerated.

**Stage 1 says images are missing**: for CCT20 the archive is incomplete, so
download it again; for Serengeti, re-run stage 0.

**Out of memory in stage 1**: lower `batch_size` in `VisionCfg` in
`sunsched/config.py`, for example to 8. Batch size does not change any result, so
this is safe even during the test run.

**Windows: stage 4 or 5 hangs at start**: try `--workers 1` and send the output.

**Anything else**: send the last 30 lines of output.
