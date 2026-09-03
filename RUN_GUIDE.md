# Running EcoExit on your PC — a guide for a friend

This is the full EcoExit system: a trained neural network, a solar/battery
simulator, and a controller that decides how to run the network to stretch a
battery as far as possible. This guide gets it running on **your** computer
from a completely fresh start. No GPU needed — everything runs on CPU.

Total time: about **10–15 minutes** for the quick version, or **1–2 hours**
for the full version. Total download: about **170 MB** (a public image
dataset called CIFAR-100).

---

## 1. Install Python

You need Python 3.10, 3.11, or 3.12. (3.13/3.14 also work but some packages
are newer there — if you hit install errors, prefer 3.11.)

- **Windows:** download from [python.org/downloads](https://www.python.org/downloads/).
  During install, **check the box "Add Python to PATH"**.
- **Mac:** `brew install python@3.11` (if you have [Homebrew](https://brew.sh)),
  or download from python.org.
- **Linux:** almost certainly already installed (`python3 --version` to check).

Check it worked by opening a terminal (Command Prompt / PowerShell on
Windows, Terminal on Mac/Linux) and running:

```bash
python --version
```

If that says "command not found", try `python3 --version` instead — on
Mac/Linux the command is usually `python3`, not `python`. Use whichever one
works for every command below.

## 2. Get the project folder

Copy the whole `EcoExit-Model` folder onto your computer (USB drive, cloud
link, git clone — whatever your friend sent it via). Open a terminal and
navigate into it:

```bash
cd path/to/EcoExit-Model
```

(On Windows, you can type `cd ` then drag the folder into the terminal
window to auto-fill the path.)

## 3. Create a virtual environment (recommended, not required)

This keeps EcoExit's packages separate from anything else on your machine.

```bash
python -m venv venv
```

Activate it:

- **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
- **Windows (cmd.exe):** `venv\Scripts\activate.bat`
- **Mac/Linux:** `source venv/bin/activate`

You'll know it worked because your terminal prompt now starts with `(venv)`.
Every command below assumes this is active — if you close the terminal and
reopen it later, just re-run the activate command.

## 4. Install the required packages

```bash
pip install -r requirements.txt
```

This installs PyTorch (the neural network library), torchvision (for the
dataset), NumPy, SciPy, and Matplotlib. It's about 200–300 MB and takes a
few minutes.

If `pip install torch` fails or seems to hang forever, install the CPU-only
build directly instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib
```

## 5. Run everything, one command

```bash
python run_all.py --quick
```

That's it — this runs the whole pipeline (train → profile energy → build
solar traces → run the full comparison → write the report) in sequence.
The first run will pause for a minute or two while it downloads CIFAR-100
(~170 MB) automatically — that's normal, just let it finish.

You'll see five stages print their own progress. **`--quick` is a fast
laptop preset** (fewer training epochs, a shorter solar trace, skips the two
most expensive baseline controllers) — good for confirming everything works
and looking at the results. Expect **10–15 minutes** total on an ordinary
laptop.

### Want the full version instead?

Once the quick run works, you can run the real thing:

```bash
python run_all.py
```

This trains longer, runs a 45-day trace per site, and includes every
baseline controller (including the two reinforcement-learning ones, which
are the slowest part). Expect **1–2 hours** depending on your CPU. There's
no harm in doing `--quick` first to make sure nothing is broken, then
running the full version afterward — it just overwrites the same output
files with better numbers.

## 6. Look at the results

Everything lands in the `results/` folder:

- **`results/REPORT.html`** — open this in your browser. It's the whole
  story: training curves, the energy-model finding, every experiment, every
  chart, one scrollable page. **Start here.**
- `results/REPORT.md` — the same report as plain text/Markdown.
- `results/figures/*.png` — every chart on its own, if you want one for a
  slide deck.
- `results/tables/*.csv` — every number, if you want to re-plot something
  yourself in Excel or Python.
- `artifacts/backbone.pt` — the trained model checkpoint itself.

## Troubleshooting

**"No module named torch" (or numpy/scipy/matplotlib)**
You skipped step 4, or the virtual environment isn't active. Re-run:
`pip install -r requirements.txt`

**It's stuck at "downloading cifar-100-python.tar.gz"**
That's a real ~170 MB download from a university server that's sometimes
slow — it's not frozen, just slow. Leave it a few minutes. If it fails
outright (a red error mentioning a URL), you're probably behind a firewall
that blocks it; try a different network.

**"CUDA out of memory" or anything mentioning a GPU**
This project runs entirely on CPU on purpose — you don't have a GPU issue
because it never touches one. If you see a GPU-related error, something
unusual is happening; feel free to flag it, but it isn't expected.

**A script fails partway through**
Each stage writes its files before the next stage starts, and each script
can also be re-run on its own — you don't have to restart from scratch:

```bash
python scripts/01_train_backbone.py --quick
python scripts/02_profile_energy.py
python scripts/03_build_traces.py --days 10
python scripts/04_run_experiments.py --skip-rl
python scripts/05_make_report.py
```

(Drop `--quick`, `--days 10`, and `--skip-rl` for the full versions.) If one
of these prints a Python error (a "Traceback"), copy the last ~15 lines and
send them back — that's the part that says what actually went wrong.

**Everything ran but I want to re-generate just the report**
`python scripts/05_make_report.py` — it only reads files the other stages
already wrote, so it's instant and safe to re-run any time.

## What am I actually looking at?

In one sentence: a small camera-vision neural network that can run at three
resolutions and stop early at three different depths, controlled by a
system that decides — every time step, based on the battery's charge and a
forecast of incoming solar power — which resolution and depth to use, or
whether to skip the frame entirely to save energy. The report compares this
controller against several controllers from real published papers
(re-implemented here, not strawmen) on five simulated solar climates, and
measures not just accuracy and energy but battery lifespan and the carbon
footprint of the whole physical device over its deployment life.
