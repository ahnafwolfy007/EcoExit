# EcoExit v2: SunSched

**Scheduling inference in time on solar camera traps.**

Existing energy-aware inference controllers decide *how much* compute a frame
gets, given the battery. SunSched starts from a measurement that makes this the
wrong question on a solar camera trap. In the Caltech Camera Traps data, most
animal captures arrive at night, when solar harvest is zero, and few arrive at
midday, when most of the energy does. The battery that covers that gap wears out
mainly by *sitting charged*, especially in a hot enclosure. SunSched therefore
decides *when* each frame's compute happens:

1. **Triage now.** Every capture is classified immediately by a cheap early exit
   on a low-power path.
2. **Refine when the sun pays.** Frames worth a better label are stored and
   refined in batches with the full network. One processor wake serves the whole
   batch, and the batch uses only energy above tonight's reserve.
3. **Do not bank the sun.** The battery is charged to a risk-controlled reserve
   (an Adaptive Conformal Inference bound on what the coming night needs), not to
   100%.

**Status: built, not yet run on real data.** The code passes its unit tests and a
synthetic end-to-end run. Whether SunSched beats existing approaches is decided by
`pipeline/06_kill_test.py` on the real data. That test is pre-registered: its
margins were fixed before any results existed.

## Quick start

```bash
pip install -r requirements.txt
python -m sunsched.tests                  # seconds; must pass
python pipeline/00_fetch_datasets.py      # ~6.6 GB, resumable
python run_sunsched.py --skip-fetch       # full pipeline -> outputs/results/REPORT.md
```

Full instructions are in [RUN_GUIDE.md](RUN_GUIDE.md). Every data source and link
is in [DATASETS.md](DATASETS.md).

## What is and is not claimed

**Built on established work (not claimed as new):** early-exit networks;
energy-aware per-frame early exit (Bullo et al., MLSP 2023 / arXiv 2411.02471);
lazy, deadline-aware scheduling on harvesting nodes (Moser et al., 2006/2007);
charge limiting from predicted harvest (IEICE Trans. E103-D, 2020); rainflow-based
battery ageing costs (Xu et al., IEEE TPS 2018); Adaptive Conformal Inference
(Gibbs and Candès, NeurIPS 2021). Each one appears here either as a baseline or as
a component.

**Candidate contribution, to be confirmed by the kill test:** the measured
mismatch between when events arrive and when energy arrives on real co-located
data. Also:

- a triage-now, refine-later action space for early-exit inference;
- a controller that combines it with calendar-ageing-aware charge control and a
  conformal night reserve;
- a benchmark that replays real camera captures against real weather.

## Layout

```
sunsched/
  config.py          every constant; ASSUMPTIONs marked and swept
  data/cct20.py      CCT20 splits, image streaming from the archive
  vision/            frozen MobileNetV3-Large trunk, exit heads, MAC counts
  env/               PVGIS weather, battery ageing, capture streams, conformal reserve
  sim/               two-tier node energy model, the slot-by-slot simulator
  policies/          baselines.py (existing approaches), ours.py (SunSched)
  eval/              metrics, bootstrap CIs, paired tests
  experiment.py      parallel job runner shared by every stage
  tests/             unit tests and a synthetic end-to-end smoke test
pipeline/            00 fetch, 01 features, 02 heads, 03 environment,
                     04 main comparison, 05 sweeps, 06 kill test, 07 report
run_sunsched.py      runs all stages in order
```

`ecoexit/`, `scripts/`, `tests/` and `run_all.py` are the superseded v1
implementation, kept only until they are removed.

## Design choices that exist to survive review

- **Real events, real images, real weather.** Each simulated camera replays its
  own captures with their real timestamps, and the classifier is scored on the
  image actually taken. Weather is PVGIS hourly irradiance and air temperature.
  Weather years are the replicates.
- **Location-disjoint evaluation.** The heads are trained on 10 cameras,
  calibrated on 1, and evaluated on 9 others.
- **Energy is not the only cost.** Battery life is computed from calendar and
  cycle ageing driven by the simulated state-of-charge and temperature traces.
- **The regime is the x-axis.** Panels are sized by harvest-to-demand ratio, so
  results say where scheduling matters and where it does not.
- **Every assumption is swept.** Node power figures, battery size, enclosure
  heating, ageing constants, deadline and charge-ceiling floor are each pushed to
  both ends of their range in `pipeline/05_sweeps.py`.
- **Paired statistics with enough replicates.** Tests pair over 9 cameras × 5
  years = 45 replicates, with Holm correction.
