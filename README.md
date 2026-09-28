# EcoExit v2: SunSched

**On a solar camera trap, battery size decides whether to classify now or later.**

In real camera-trap data most animal captures arrive at night, when solar
harvest is zero. The first full run on Caltech Camera Traps (CCT20) measured
55.7%, while the high-sun hours that deliver 54% of the energy bring only 17.9%
of captures. The battery bridges that gap, and it ages mainly by *sitting
charged*. That run also failed its own pre-registered test: the proposed
controller did not beat existing approaches.

Looking at why pointed to a regime effect, which this version tests properly:

- **Small storage (days of autonomy):** there is no spare energy at night, so the
  right move is to capture and triage now and **classify later** in daylight.
- **Large storage (weeks):** the night costs nothing, so classify now, and
  **stop charging to 100%** to spare the battery.

Because that effect was found by looking at data already used, it is a
hypothesis. **[PREREGISTRATION.md](PREREGISTRATION.md)** fixes the tests,
comparisons and margins. They are decided on a corpus nobody had looked at:
**Snapshot Serengeti**, from another continent. CCT20 is now the development
corpus.

**SunSched v2** is one rule meant to be right at every battery size. It triages
every capture, then refines a frame immediately if the energy above tonight's
conformal reserve pays for it, and otherwise defers it to daylight. It charges
only to that reserve plus a margin. With a large battery it behaves like
refine-now plus a ceiling; with a small one it behaves like deferral.

**Status: built and tested on synthetic data; not yet run on either real corpus
in this version.**

## Quick start

```bash
pip install -r requirements.txt
python -m sunsched.tests                                   # seconds; must pass
python pipeline/00_fetch_datasets.py --corpus cct20        # development data
python run_sunsched.py --corpus cct20 --skip-fetch         # develop and tune here
# commit any changes, then run the test once:
python pipeline/00_fetch_datasets.py --corpus serengeti
python run_sunsched.py --corpus serengeti --skip-fetch     # decides the hypotheses
```

Full instructions are in [RUN_GUIDE.md](RUN_GUIDE.md). Every data source and link
is in [DATASETS.md](DATASETS.md).

## What is and is not claimed

**Established work, used as baselines or components:** early-exit networks;
energy-aware per-frame early exit (Bullo et al., MLSP 2023 / arXiv 2411.02471);
lazy, deadline-aware scheduling on harvesting nodes (Moser et al., 2006/2007);
charge limiting from predicted harvest (IEICE Trans. E103-D, 2020); battery ageing
models (Xu et al., IEEE Trans. Smart Grid 2018); Adaptive Conformal Inference
(Gibbs and Candès, NeurIPS 2021).

**Claims under test (PREREGISTRATION.md):**

- **H1:** the day/night mismatch between captures and energy replicates on held-out data.
- **H2 and H3:** which strategy is right depends on battery autonomy. This is the
  primary claim.
- **H4 and H5:** SunSched v2 wins where deferral should, and is near-best
  everywhere. These decide whether it is presented as a method.
- The co-located benchmark of real captures and real weather.

## Layout

```
sunsched/
  config.py          every constant; ASSUMPTIONs marked; HypothesisCfg is pre-registered
  data/              cct20.py, serengeti.py (fixed selection rule), corpus.py
  vision/            frozen MobileNetV3-Large trunk, exit heads, MAC counts
  env/               PVGIS weather, battery ageing, capture streams, conformal reserve
  sim/               two-tier node energy model, slot-by-slot simulator
  policies/          baselines.py (existing approaches), ours.py (SunSched v2)
  eval/              metrics, bootstrap CIs, paired tests
  experiment.py      parallel runner over the autonomy x ratio grid
  tests/             26 unit tests and a synthetic end-to-end smoke test
pipeline/            00 fetch, 01 features, 02 heads + detector, 03 environment + H1,
                     04 main grid, 05 sweeps, 06 hypotheses, 07 report
run_sunsched.py      runs all stages for one corpus
PREREGISTRATION.md   the hypotheses, fixed before the test data was seen
```

`ecoexit/`, `scripts/`, `tests/` and `run_all.py` are the superseded v1
implementation, kept only until they are removed.

## Design choices that exist to survive review

- **Development and test are different corpora.** Anything may be tuned on CCT20.
  Serengeti is run once with committed code, and the run records its git commit.
- **Real events, real images, real weather.** Each simulated camera replays its own
  captures with their real timestamps, and the classifier is scored on the image
  actually taken. Weather is PVGIS hourly irradiance and air temperature; weather
  years are the replicates.
- **Location-disjoint roles** within each corpus: train, calibration and evaluation
  cameras never overlap.
- **Both regime axes are explicit.** Battery autonomy is measured in days of each
  camera's own essential demand; harvest is measured relative to demand.
- **No free energy.** Each deployment is simulated to a cyclic steady state, so a
  large battery's starting charge is not a months-long subsidy.
- **A triage signal that carries information.** Refinement gain is indexed by a
  dedicated animal-vs-empty detector, not by the species head's near-constant
  top-1 confidence.
- **Fair baselines.** `lazy_animal` gives deferral the same detector signal, and
  `ceiling_now` judges its battery relative to its own ceiling.
- **Every assumption is swept**, at both small- and large-storage cells.
- **Paired statistics** over camera × weather-year replicates, with Holm
  correction over cells.
