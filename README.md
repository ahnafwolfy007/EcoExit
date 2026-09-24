# EcoExit — solar-forecast-aware adaptive inference

A full, runnable implementation of the rebuilt EcoExit methodology: a
shadow-price controller that jointly picks input resolution, network exit
depth, and duty cycle for a vision model on a solar+battery edge node,
conditioned on a risk-calibrated harvest forecast and charged against the
battery's cycle life. See [`RUN_GUIDE.md`](RUN_GUIDE.md) if you just want to
run it; this file explains what's actually in the box.

If you have not read the gap-analysis/methodology writeup this implements,
its structure follows sections 04–13 of that document one-to-one: system
model → controller → energy/battery/carbon models → datasets → evaluation
protocol → positioning → timeline.

## Quick start

```bash
pip install -r requirements.txt
python tests/test_core.py             # verify the install, no downloads
python scripts/00_fetch_datasets.py   # ~180 MB, no account needed
python run_all.py --quick             # ~15 min, laptop-friendly
```

Then open `results/REPORT.html`. Every dataset link is in
[`DATASETS.md`](DATASETS.md); setup, flags and troubleshooting are in
[`RUN_GUIDE.md`](RUN_GUIDE.md).

**Read the Phase 0 gate first** (`scripts/06_phase0_gate.py`, stage 4 of
`run_all.py`). It decides whether the controller is doing anything at all, and
it is designed to be able to say no. Its current verdict, and why, is in
[`RUN_GUIDE.md`](RUN_GUIDE.md#current-state-as-of-this-build).

## What each claim looks like as code

| Claim | Where it lives |
|---|---|
| 1 — Carbon-optimal ≠ energy-optimal | `ecoexit/eval/metrics.py` (`battery_carbon_share`, `gco2e_per_correct_inference`) |
| 2 — Path-dependent wear price from the rainflow residual | `ecoexit/wear/rainflow.py` (`OnlineWearPricer`) |
| 3 — Learning-augmented guarantee on the conformal bound | `ecoexit/forecast/conformal.py`, `ControlCfg.trust` |
| 4 — Co-located event + harvest benchmark | `ecoexit/data/camera_traps.py`, `ecoexit/solar/pvgis.py` |

Supporting method, not claimed as contributions: energy shadow pricing
(`ecoexit/control/pricing.py`), the measured-vs-MACs energy model
(`ecoexit/energy/model.py`), and the duty-cycle knob (`ecoexit/sim/loop.py`).

## Regime and sizing

`ecoexit/sizing.py` computes the harvest-to-demand ratio, which is the
independent variable every headline figure should be plotted against. At a
fixed panel area four of five sites harvest more than the most expensive policy
could spend, which makes `static_max` correct rather than weak there; the gate
resizes each site to land inside the discretionary band before comparing.

## Project layout

```
ecoexit/
  config.py            All tunable constants in one place. Every value that
                        should eventually come from a datasheet or a real
                        measurement is commented LITERATURE.
  models/elastic.py     The multi-resolution x multi-exit backbone
                        (switchable BatchNorm + weighted multi-exit loss +
                        sandwich-rule training sampling).
  data/cifar.py          CIFAR-100 loading (the labelled-image pool only --
                        see the report's Hole H4 on why timing is separate).
  energy/model.py       Measures MACs/activation-bytes/weight-bytes/time via
                        forward hooks; DecomposedModel vs MacsOnlyProxy.
  solar/traces.py        Analytic clear-sky geometry x an AR(1) cloud
                        process -- five climates, fully offline.
  sim/stream.py          Event-sparse, crepuscular-weighted frame-value
                        stream synthesised over the solar clock.
  forecast/conformal.py  Clear-sky persistence forecaster + Adaptive
                        Conformal Inference calibration.
  battery/model.py       SoC dynamics (asymmetric efficiency, self-discharge)
                        + rainflow cycle counting for wear.
  control/pricing.py     The shadow-price controller: concave envelope,
                        directional water-filling, the confidence-threshold
                        fast loop.
  control/baselines.py   Reimplemented competitors: ePerceptive's reactive
                        LUT, HarvSched-style Q-learning, a Bullo-style
                        monotone MDP, and the perfect-foresight oracle DP.
  sim/loop.py             The closed-loop, slot-by-slot simulator every
                        policy runs through identically.
  eval/metrics.py         CTU, value of forecast, percent-of-oracle, etc.

scripts/
  01_train_backbone.py    Train the elastic backbone on CIFAR-100.
  02_profile_energy.py    Measure real energy, compare vs. MACs-only proxy,
                        precompute per-image outcomes + confidence calibration.
  03_build_traces.py      Build the five site traces + conformal forecasts.
  04_run_experiments.py   The full policy x site comparison, plus all four
                        headline experiments and the ablations.
  05_make_report.py       Assemble everything into results/REPORT.{md,html}.

run_all.py                Runs 01-05 in order.
results/                  Every table, figure, and the final report.
artifacts/                The trained checkpoint and precomputed arrays.
```

## Design choices worth knowing about (and why)

**The elastic backbone is intentionally small** (~100K parameters, three
tiny ResNet-style stages) so the whole pipeline trains on a CPU in minutes.
Its *raw* measured energy is therefore far below a production vision
model's. `scripts/02`'s MACs-only-vs-decomposed mismatch finding uses those
raw numbers as-is (the finding is about *ranking disagreement*, which
doesn't depend on absolute scale); the closed-loop simulation in
`scripts/04` rescales per-inference energy by a stated
`EnergyCfg.system_scale_factor` to land in a believable Jetson/Pi-class
wattage range. Both facts are stated in `results/REPORT.md`, not hidden.

**The solar panel is deliberately undersized** relative to the (now
realistically-scaled) workload — see the comment on `SolarCfg.panel_area_m2`
in `ecoexit/config.py`. A generously provisioned panel makes any adaptive
controller largely unnecessary; the point of this project is a controller
for a genuinely energy-scarce deployment, so the simulation is built to
actually feel that scarcity rather than comfortably avoid it.

**Event timing is synthetic; the images underneath are real.** A real
camera-trap-plus-solar-plus-battery co-located dataset was out of scope for
a course project (Hole H4). `ecoexit/sim/stream.py` synthesises *when*
interesting things happen (Poisson-cluster, crepuscular-weighted, matching
real camera-trap literature); every classification outcome credited during
simulation is a real CIFAR-100 test image run through the real trained
model.

**The pre-capture value estimate is not the ground truth.** Every online
controller (ours, the reactive LUT, Q-learning, the MDP) decides whether to
wake using `ecoexit.sim.stream.expected_value_curve` — the *historical*
time-of-day expected frame value — never the realised value of the specific
frame about to be captured, which no real system could know in advance. The
perfect-foresight oracle is the one policy allowed to see the true future,
because that is what "oracle" means.

**Baselines are real reimplementations, named after their papers**, not
strawmen: `ReactiveLUT` mirrors ePerceptive's charge-time-indexed table,
`HarvSchedQLearning` mirrors HarvNet/HarvSched's tabular RL over
(SoC, harvest, time), `MonotoneMDPController` mirrors Bullo et al.'s
harvest-state MDP (deliberately without time-of-day awareness, matching
that paper's own real limitation). Every simplification relative to the
original paper is documented in that class's docstring in
`ecoexit/control/baselines.py`.

## Re-running just one stage

Every script under `scripts/` can be run on its own once its dependencies
exist (each prints what it needs). Useful during iteration:

```bash
python scripts/04_run_experiments.py --skip-rl   # skip the two slow RL baselines
python scripts/03_build_traces.py --days 10       # shorter traces for a fast check
python scripts/05_make_report.py                  # just rebuild the report from existing tables
```
