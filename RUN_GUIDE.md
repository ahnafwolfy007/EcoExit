# Running EcoExit

This is the full EcoExit system: a multi-exit neural network, a solar/battery
simulator, and a controller that decides — every minute of simulated time —
whether to wake, at what resolution, and how deep into the network to run,
trading accuracy against energy and battery wear.

Everything runs on **CPU**. No GPU needed for anything in this guide.

- **Quick check**: ~15 minutes, ~180 MB download.
- **Full run**: 1–3 hours depending on your CPU.

---

## 1. Install Python

Python **3.10, 3.11 or 3.12**. (3.13+ usually works; if you hit install
errors, use 3.11.)

- **Windows**: [python.org/downloads](https://www.python.org/downloads/) —
  tick **"Add Python to PATH"** during install.
- **Mac**: `brew install python@3.11`, or python.org.
- **Linux**: almost certainly already there.

```bash
python --version
```

If that says "command not found", try `python3` instead, and use `python3` for
every command below.

## 2. Get the project and open a terminal in it

```bash
cd path/to/EcoExit-Model
```

On Windows you can type `cd ` and then drag the folder into the terminal.

## 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows PowerShell**: `.\venv\Scripts\Activate.ps1`
- **Windows cmd.exe**: `venv\Scripts\activate.bat`
- **Mac/Linux**: `source venv/bin/activate`

Your prompt should now start with `(venv)`. If you close the terminal, re-run
the activate command before doing anything else.

## 4. Install packages

```bash
pip install -r requirements.txt
```

If `torch` fails or hangs, install the CPU-only build directly:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 5. Check the install

```bash
python tests/test_core.py
```

These are unit tests with no downloads and no training — they check the wear
model, the pricing bisection and the sizing maths. **All of them should pass.**
If they do not, stop here and send the output; nothing downstream will be
trustworthy.

## 6. Download the data

```bash
python scripts/00_fetch_datasets.py
```

About 180 MB: CIFAR-100 (the images), the Caltech camera-trap metadata (real
capture timestamps), and cached PVGIS irradiance for each site. Nothing needs
an account or an API key.

Add `--all` to also pull Snapshot Serengeti (another 188 MB) — only needed for
the large-scale and multi-season experiments.

Every source and direct link is listed in [DATASETS.md](DATASETS.md), including
what to do if one is blocked on your network.

## 7. Run it

```bash
python run_all.py --quick
```

That runs the whole pipeline in order: train → profile → build traces → **the
gate** → full comparison → report. Expect **~15 minutes**.

Once that works, the real thing:

```bash
python run_all.py
```

And with real irradiance and real camera-trap event timing:

```bash
python run_all.py --real-data
```

---

## The one result that matters

Stage 4 is the **Phase 0 gate**. Read its output before anything else.

The question it answers is whether the controller is actually doing anything.
There is a specific failure this project has already hit once: if the planner
hands the controller more energy budget than it can possibly spend, the price
collapses to zero, the controller degenerates into "always run the biggest
model", and every result downstream becomes a restatement of that baseline
while still looking like a table of numbers.

The gate checks four things and prints `PASS` or `FAIL` for each:

0. **Is carbon measurable?** The policy can only move one component of
   lifecycle carbon — the battery, by wearing it out faster or slower. Panel
   and board are fixed the moment the node is deployed. If the battery is a
   rounding error next to the board, then "carbon-normalized utility" is just
   total value divided by a constant, every policy ranks exactly as it does on
   raw value, and the carbon inversion cannot appear however good the
   controller is.
1. **Is the price live?** Non-zero and varying, not pinned at zero.
2. **Does the regime bind?** Daily solar harvest has to land *between* the
   always-asleep floor and the always-awake-at-maximum ceiling. Above the
   ceiling there is simply no decision to make and "always maximum" is the
   correct answer — not a weak baseline.
3. **Does the controller beat `static_max`?** Across seeds, with a paired
   statistical test.

```bash
python scripts/06_phase0_gate.py
python scripts/06_phase0_gate.py --profile mcu     # different node hardware class
```

Exit code `0` means pass, `2` means the gate ran correctly and the verdict is
fail. **A FAIL is a real answer, not a crash.** The script prints which
condition failed and what to check.

### Current state, as of this build

Conditions 1 and 2 pass. Conditions 0 and 3 fail, and 0 explains 3:

- With the default `sbc` hardware profile, the battery is **0.06%** of
  lifecycle carbon at the wear the simulation actually produces (1.5% even at
  a full replacement), against a 22 kgCO2e board. So the carbon metric cannot
  separate policies, and it reduces to raw captured value.
- On raw captured value, `static_max` wins, because a brownout is free in the
  current energy model: an unaffordable wake silently falls back to idle at no
  cost. That makes "always try the biggest model" a genuinely strong policy,
  and the proposed controller's voluntary reserve a pure loss.

Both are properties of the *parameters and the energy model*, not bugs in the
controller — and both need a decision before the carbon claim can be tested.
The `--profile` flag exists to explore the first; the second needs a cost for
over-committing.

Results land in `results/tables/gate_verdict.json`, `gate_sizing.csv`,
`gate_runs.csv` and `gate_tests.csv`.

---

## Where everything ends up

| Path | What it is |
|---|---|
| `results/REPORT.html` | The whole story in one scrollable page. **Start here.** |
| `results/tables/gate_verdict.json` | Pass/fail on the three gate conditions. |
| `results/tables/gate_sizing.csv` | Per site: harvest-to-demand ratio and regime. |
| `results/tables/*.csv` | Every number, for re-plotting. |
| `results/figures/*.png` | Every chart on its own. |
| `artifacts/backbone.pt` | The trained model checkpoint. |
| `artifacts/traces/` | Per-site solar traces and forecasts. |

After a run, this checks the artifacts for the leakage problems the pipeline is
designed to avoid:

```bash
python tests/test_pipeline.py
```

---

## Running stages individually

Every stage writes its files before the next begins, so you can re-run any one
of them without starting over:

```bash
python scripts/00_fetch_datasets.py
python scripts/01_train_backbone.py --quick
python scripts/02_profile_energy.py
python scripts/03_build_traces.py --days 10
python scripts/06_phase0_gate.py --quick
python scripts/04_run_experiments.py --skip-rl
python scripts/05_make_report.py
```

Drop `--quick`, `--days 10` and `--skip-rl` for full versions.

Useful flags:

| Flag | Effect |
|---|---|
| `03 --source pvgis` | Real measured irradiance instead of the analytic generator. |
| `03 --events camera_trap` | Real capture timestamps instead of the synthetic process. |
| `06 --seeds 5` | More seeds, tighter confidence intervals. |
| `06 --ratio 0.6` | Target a scarcer regime when sizing the panels. |
| `06 --no-resize` | Keep the original panel size, to reproduce the original tie. |

---

## Troubleshooting

**"No module named torch" / numpy / scipy**
Step 4 was skipped, or the virtual environment is not active.

**Stuck downloading `cifar-100-python.tar.gz`**
It is a slow university server, not a hang. Give it a few minutes. If it fails
outright, you are probably behind a firewall — try another network, or download
it manually per [DATASETS.md](DATASETS.md).

**Camera-trap metadata will not download**
Run with `--events synthetic` (the default). The pipeline works end to end
without it; you only lose the real-event-timing experiments.

**PVGIS unreachable**
Leave the irradiance source as `analytic`. Same deal — everything runs, you
lose only the real-irradiance claim.

**Anything mentioning CUDA or a GPU**
This pipeline never touches a GPU. If you see a GPU error something unusual is
happening; send the message.

**The gate says FAIL**
That is the pipeline working. Read the diagnostics it prints — it names which
of the three conditions failed and what to change. It is meant to be able to
say no.

**A script raises a traceback**
Send the last ~15 lines. That is the part that says what actually went wrong.

---

## What is actually being simulated

A solar-powered camera node in the field with a small panel and a small
battery. Animals trigger the camera in bursts, mostly at dawn and dusk, and
most frames turn out to be empty. The node cannot afford to run its largest
model on every frame, so a controller continuously decides how much compute
each frame is worth, given how much charge is left and how much sun is
forecast.

The twist the project is built around: for an off-grid node, **minimizing
energy is the wrong objective**. Operational carbon is essentially zero — the
sun is free — so what the policy actually controls is the *embodied* carbon of
the hardware, and the only part of that a control policy can change is how fast
it wears the battery out. Battery wear depends on how deeply the charge cycles,
not on how many joules pass through. So the policy that maximizes work per
joule is not the policy that minimizes lifetime carbon, and the controller here
prices both.
