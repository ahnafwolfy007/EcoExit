# Pre-registration: storage size decides the schedule

Written and committed **before any Snapshot Serengeti data was downloaded or
seen**. The hypotheses, comparisons, margins and pass rules below are fixed.
`pipeline/06_hypotheses.py` implements them exactly. Anything reported beyond
them must be labelled exploratory.

## Background: what the development data showed

The first full run was on CCT20 (Caltech Camera Traps, 9 held-out cameras). It
failed its own pre-registered kill test: the proposed controller did not beat
existing approaches. Looking at that data *after* the fact suggested two
regularities:

1. **Small storage (under about 3 days of battery autonomy): deferral wins.**
   Storing uncertain frames and classifying them in daylight beat classifying
   everything immediately (`lazy_defer` 0.271 vs `always_now` 0.241 value score
   at 1 day, harvest ratio 1.0).
2. **Large storage (weeks or more): a charge ceiling wins.** Capping the charge
   roughly doubled battery life (14.6 vs 7.7 years at ratio 2.0) at no cost in
   value.

Both were found by looking at data the original test had already used, so they
are hypotheses, not results. This document tests them on data nobody has looked at.

## Data roles

| Role | Data | Use |
|---|---|---|
| Development | CCT20: all 20 cameras, including the 9 `trans_test` cameras already seen | Design, debugging and tuning. Anything goes. |
| **Test** | **Snapshot Serengeti**, one season. Official `val` locations only, never used in development | Run once with the frozen configuration. These hypotheses are decided here. |

On Serengeti, heads are trained on official `train` locations. They are
calibrated on 3 `val` cameras and evaluated on the remaining `val` cameras. The
camera selection rule is in `sunsched/data/serengeti.py` and was fixed before
download.

**Freezing.** Every constant used on the test corpus is the value committed in
`sunsched/config.py` at the commit that runs it. Stage 4 writes that commit hash
and the full configuration to `run_manifest.json`. Changing a constant after
seeing Serengeti results and re-running invalidates the test. It must then be
reported as a second, exploratory run.

## Experimental grid

Replicates are camera × weather year (PVGIS 2011–2015): for example, 10 cameras
× 5 years = 50 paired replicates per cell. Each simulation runs twice and only
the second pass is scored (cyclic steady state), so the initial state of charge
gives no free energy.

- **Battery autonomy**: 0.5, 1, 2, 3, 5, 10, 30, 100 days. That is the battery
  capacity divided by the camera's own mean daily essential energy (sleep,
  capture, triage).
- **Harvest-to-demand ratio**: 0.75, 1.0, 1.5, 2.0, 3.0.

## Hypotheses

The same margin constants are used for every hypothesis. They are in
`HypothesisCfg` in `sunsched/config.py`.

**H1: the mismatch replicates.** On the test corpus, at least **40%** of animal
captures arrive with the sun below the horizon, and the hourly correlation
between capture share and solar-energy share is **negative**. Computed over every
camera in the chosen Serengeti season, from metadata.

**H2: deferral pays when storage is small.** Consider the 9 cells with autonomy
∈ {0.5, 1, 2} days and ratio ∈ {1.0, 1.5, 2.0}. In each cell, `lazy_defer`'s value
score exceeds `always_now`'s by at least **0.02** (mean paired difference), and a
paired two-sided Wilcoxon test gives p < 0.05 after Holm correction across the 9
cells. **H2 holds if this is true in at least 5 of the 9 cells.**

**H3: a charge ceiling pays when storage is large.** Consider the 6 cells with
autonomy ∈ {30, 100} days and ratio ∈ {1.0, 1.5, 2.0}. In each cell,
`ceiling_now`'s battery life is at least **1.15×** `always_now`'s (geometric mean
of paired ratios, Wilcoxon on log life, Holm across the 6 cells, p < 0.05), with
its value score no more than **0.01** lower. **H3 holds if this is true in at
least 4 of the 6 cells.**

**H2 and H3 together are the primary claim**: which strategy is right depends on
battery autonomy.

**H4: SunSched v2 beats immediate classification when storage is small.** Same
test as H2 with `sunsched` in place of `lazy_defer`.

**H5: SunSched v2 is near-best everywhere.** For every cell with ratio ≥ 1.0,
let *V* be the best mean value score among the baselines (`always_now`,
`triage_only`, `ee_now`, `ceiling_now`, `lazy_defer`, `lazy_animal`). Let *L* be
the longest mean battery life among the baselines whose value is within 0.01 of
*V*. SunSched v2 is near-best in that cell if its mean value ≥ *V* − 0.01 and its
mean life ≥ 0.95 × *L*. **H5 holds if SunSched v2 is near-best in at least 75% of
those cells.** This is descriptive, with no p-value; it is a claim about means.

## What each outcome means

| H1 | H2 | H3 | Claim that can be made |
|---|---|---|---|
| ✓ | ✓ | ✓ | The storage-dependent regime map, on held-out data from another continent. |
| ✓ | ✓ | ✗ | Deferral pays at small storage; the ceiling result does not generalise. |
| ✓ | ✗ | ✓ | Charge ceilings pay at large storage; deferral does not generalise. |
| ✓ | ✗ | ✗ | Only the measurement. |
| ✗ | – | – | The day/night mismatch is specific to CCT20. |

H4 and H5 decide only whether SunSched v2 is worth presenting as a method. They
do not affect the regime claim.

## Known limitations, stated in advance

- No hardware. Node power figures are datasheet-typical assumptions, swept in
  `pipeline/05_sweeps.py`.
- Camera locations are region-level: Serengeti is one coordinate for the park,
  CCT20 one for its region.
- Captures from different calendar years at one camera are laid onto one
  simulated year.
- The battery-ageing constants (Xu et al. 2018) must be verified against the
  paper before any lifetime number is reported.
- The frozen trunk is a weak classifier on unseen cameras (CCT20: 22–33% top-1).
  Absolute value scores are low. The hypotheses concern differences between
  policies, which share the classifier.
