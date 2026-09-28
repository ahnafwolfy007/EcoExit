# Running SunSched

This guide takes a fresh machine to a finished `REPORT.md`. Everything runs on a
CPU; no GPU and no hardware are needed. A GPU is optional and only shortens
stages 1 and 2 -- see [5b](#5b-optional-stages-1-and-2-on-a-gpu).

> The folders `ecoexit/`, `scripts/`, `tests/` and `run_all.py` are the **old v1
> code**. Ignore them. Everything below uses `sunsched/`, `pipeline/` and
> `run_sunsched.py`.

## What you need

| | |
|---|---|
| Python | 3.10, 3.11 or 3.12 (3.13+ usually works; use 3.11 if installs fail) |
| Disk | about **10 GB** free: 6.5 GB image archive, about 1 GB of cached features, plus outputs |
| RAM | 8 GB or more |
| CPU | any; more cores make the simulation stages faster |
| GPU | not needed. An NVIDIA GPU shortens stages 1-2 only, via `--device cuda` |
| Network | needed for the downloads and for PVGIS on first use; later runs use the cache |

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

If installing torch fails, install the CPU build explicitly, then run the line above again:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 2. Check the install (seconds)

```bash
python -m sunsched.tests
```

All 18 tests must pass. They use synthetic data only. Optionally, run the same
plus a 2–5 minute end-to-end run of the later pipeline stages on a synthetic world:

```bash
python -m sunsched.tests --slow
```

If anything fails here, stop and send the output back. Nothing later can be trusted.

## 3. Download the data

```bash
python pipeline/00_fetch_datasets.py
```

About 6.6 GB. How long it takes depends on your connection. The download
**resumes** if it stops, so just run the same command again. Every link is in
[DATASETS.md](DATASETS.md), including manual `curl` commands.

Do not extract `eccv_18_all_images_sm.tar.gz`; the pipeline reads it as it is.

## 4. Quick run first

```bash
python run_sunsched.py --skip-fetch --quick
```

This uses a subset of the images, 2 weather years and a 90-day season, and exists
to show that everything works on your machine. Most of its time is feature
extraction, which still has to read the whole 6.5 GB archive once. The numbers it
produces are **not** results.

## 5. Full run

```bash
python run_sunsched.py --skip-fetch
```

Rough timing on an 8-core laptop CPU. Your machine will differ.

| Stage | What it does | Time |
|---|---|---|
| 1 | Stream all 57,864 images through the frozen network once | 30–90 min |
| 2 | Train 6 small exit heads, calibrate them | 5–15 min |
| 3 | Fetch/cache PVGIS weather, build capture streams | 1–5 min |
| 4 | Main comparison: 10 policies × 9 cameras × 5 weather years × 3 regimes | 5–20 min |
| 5 | Frontiers, regime curves, sensitivity (about 11,000 simulations) | 30–90 min |
| 6 | Kill test | seconds |
| 7 | Report | seconds |

For a first pass, `--skip-sweeps` skips stage 5 (the kill test and report still run).

## 5b. Optional: stages 1 and 2 on a GPU

```bash
python run_sunsched.py --skip-fetch --device cuda
```

`--device` takes `cpu` (the default), `cuda`, or `auto` (CUDA when present).
`cuda` fails loudly if this venv holds the CPU-only torch build, rather than
silently falling back.

Only stages 1 and 2 use it. Stages 3-7 are numpy simulation across worker
processes and ignore it entirely, so plan for roughly half the wall-clock at
best. Stage 1 also spends much of its time decoding JPEGs and reading the
archive, which no GPU helps with.

**This cannot move a result.** The one energy input measured from the network is
its MAC count, counted from layer shapes in `sunsched/vision/cost.py`, and shapes
do not depend on the device; the node model then multiplies them by an assumed
energy per MAC. The MAC counts are bit-identical on CPU and CUDA. TF32 is turned
off (`sunsched/vision/device.py`) so cached features stay as close to a CPU run as
the hardware allows: the measured gap is ~1e-8, about 500x below the float16 step
the features are stored at. The device used is recorded in
`outputs/artifacts/features/meta.json`.

> This is the one place v2 differs from v1, where `scripts/02_profile_energy.py`
> timed the network on a CPU and fed those wall-clock timings into the energy
> model. There the device was a scientific choice; here it is not.

If a stage fails, fix the error and resume from it instead of starting over:

```bash
python run_sunsched.py --skip-fetch --from 4
```

Each stage can also be run on its own: `python pipeline/04_run_experiments.py`,
and so on. Every script takes `--quick`, `--workers N`, `--out`, `--data`.

## 6. Results

Everything lands in `outputs/results/`:

| File | What it is |
|---|---|
| `REPORT.md` | The whole story, one page. **Start here.** |
| `tables/kill_test.json` | The verdict: does SunSched beat the strongest existing approaches? |
| `tables/summary.csv` | Mean and 95% CI of every metric, per policy and regime |
| `tables/runs.csv` | One row per simulated deployment |
| `tables/frontier.csv`, `regime.csv`, `sensitivity.csv` | Stage 5 sweeps |
| `tables/diel_mismatch.csv`, `environment_summary.json` | When animals arrive versus when the sun does |
| `tables/accuracy_grid.csv` | Classifier accuracy at each exit and resolution |
| `figures/*.png` | All figures |

`outputs/artifacts/` holds cached features and model outputs. It is large and
regenerable; do not commit it.

**What to send back:** zip `outputs/results/` and send it, together with the
terminal output of the run.

## How to read the kill test

`run_sunsched.py` prints `kill test verdict: PASS` or `FAIL` at the end.

- **PASS**: SunSched beat per-frame energy-aware early exit, the same with a
  charge ceiling, and lazy deferral, by margins that were fixed before any
  results existed.
- **FAIL**: it did not. That is a valid result, not a crash; stage 6 exits with
  code 2 by design. It means the claim cannot be made as stated. The frontier and
  sensitivity tables then show whether a narrower claim still holds.

## Troubleshooting

**`No module named sunsched`**: run the commands from the `EcoExit-Model` folder
itself, not from inside `pipeline/`.

**PVGIS unreachable** (stage 0 or 3): retry later; the service is sometimes busy.
If it stays down, `--solar analytic` runs everything on synthetic weather. The
report is then marked synthetic, so say so.

**The image download stopped**: run `python pipeline/00_fetch_datasets.py --only images`
again; it continues where it stopped.

**Stage 1 says images were "not found in the archive"**: the archive is
incomplete. Delete it and download it again.

**Out of memory in stage 1**: lower `batch_size` in `VisionCfg` in
`sunsched/config.py` (for example to 8).

**Windows: stage 4 or 5 hangs at the start**: try `--workers 1` to rule out
multiprocessing, and send the output if it then works.

**Anything else**: send the last 30 lines of output. That is where the actual
error is.

## What is being simulated

A solar-powered camera trap with a small panel and battery, replaying real
captures from 9 real cameras. Every capture is triaged immediately by a cheap
early exit. Frames worth a better label are either classified at once (the
existing approaches) or stored and classified later with the full network in the
daytime surplus (SunSched). The battery ages with its state of charge and
temperature, so SunSched also avoids keeping it full, charging it only to a
risk-controlled reserve.
