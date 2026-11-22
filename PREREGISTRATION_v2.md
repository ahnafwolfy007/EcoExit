# Pre-registration v2: one controller for every battery size

Written and committed **before any data of either test corpus below was
downloaded or seen**. The controller, its constants, the comparisons, the
margins and the pass rules are fixed here. `analysis/score_v2.py`, committed
with this file, implements them exactly. Anything reported beyond them is
labelled exploratory. `PREREGISTRATION.md` (v1) and its verdicts stand
unchanged; nothing below re-tests or replaces them.

## Why a second pre-registration

The v1 test on Snapshot Serengeti S10 (commit 0b3492b) gave: H1 failed (9.9%
of animal captures at night), H2 held (deferral pays at small storage, 9 of 9
cells), H3 failed (the charge ceiling lengthened battery life but cost
0.056–0.139 value). H4 and H5, about SunSched v2, were not tested because the
paper used another simulator. The proposed controller of the paper (SunSched
v1) had already lost to fixed rules on the development data.

Those verdicts are final. This document registers a **new controller**,
SunSched v3, designed afterwards on the development corpus only, and tests it
once on **two new corpora** that no part of this project has loaded.

## Data roles

| Role | Data | Use |
|---|---|---|
| Development | CCT20 (20 Caltech Camera Traps locations) | Design of v3, five iterations, all results public in the repo |
| Seen, not used | Snapshot Serengeti S10 (the v1 test) | Only its camera list, to exclude those cameras from test B |
| **Test A** | **Caltech Camera Traps, cameras outside CCT20** (`--corpus cct_new`) | Same region as development, unseen cameras |
| **Test B** | **Snapshot Serengeti S11, cameras not used by the S10 test** (`--corpus serengeti11`) | Another continent and a day-active fauna |

Selection rules are code, committed with this file: `sunsched/data/cct_new.py`
and `sunsched/data/serengeti.py` (`select`, with `serengeti_excluded_cameras`).
Both evaluate on official `val` locations only, calibrate on the next 3
`val` locations in the same ranking, and train the heads on official `train`
locations. Each needs **at least 10 evaluation cameras**; the image budget
rises in 5,000-image steps until 10 fit. Ten, not six, because the camera is
the unit of analysis and an exact Wilcoxon test on six camera means cannot
reach p < 0.016 one-sided.

**Freezing.** v3 and every constant are the values at the commit that adds
this file. Stage 4 records the commit hash in the run manifest. A constant
changed after either test corpus is seen invalidates that test; any re-run is
reported as exploratory.

## The controller under test (SunSched v3)

`sunsched/policies/v3.py`, policy name `sunsched3`. Every uncertain frame
(triage confidence below `lazy_theta` = 0.9) is deferred; the controller
decides when to wake the refinement processor and how full to keep the
battery. Four mechanisms, each taken from a deployed system:

1. **Coalesced dispatch** (Android Doze maintenance windows, TCP Nagle,
   economic order quantity): a batch is dispatched only when its refinement
   energy is at least the wake energy, unless a frame nears its deadline,
   energy would be curtailed, or it is the daily noon window or the last
   daylight chance before a frame expires. When a wake costs less than two
   refinements, frames are refined at once (TCP_NODELAY).
2. **PV-overflow spending** (solar diverters, home energy managers): in
   daylight, the controller spends what is stored above tonight's reserve,
   or, if larger, the harvest forecast for the rest of the day (median of the
   last three days at the same clock times) that the battery could not store
   under the level in force; at night, only what the rest of the night does
   not need.
3. **Reservoir hold** (reservoir sizing and operation, charge-controller
   re-bulk): the charge is held at floor + max(sequent-peak storage
   requirement, tonight's conformal reserve) + 10% of capacity, never below
   `min_ceiling`. The sequent-peak requirement (Thomas & Burden 1963) is the
   largest cumulative deficit of demand (essential load plus the refinement
   the node asked for) over stored harvest seen so far. The hold engages after
   three consecutive surplus days. A day is surplus if the battery reached the
   level in force, or if it ended daylight at or above the hold level and held
   at least as much at sunrise as the day before. The hold is released only
   when the sunrise energy falls below tonight's reserve.
4. **Earliest deadline first** within a batch.

No constant is tuned per cell. The constants it uses are the baselines'
(`lazy_theta`, `lazy_soc_keep`, `ceiling_margin`, `min_ceiling`,
`batch_interval_min`, `deadline_h`), plus a 3-day history. Earlier
development settings exist in the code as named variants
(`sunsched3_iter1` ... `sunsched3_iter4`, `sunsched3_rule`); none is part
of this test.

### How v3 was developed and selected (CCT20 only)

| Iteration | Commit | Change | Near-best cells |
|---|---|---|---|
| 1 | 66e13b8 | Coalesced dispatch; minimum-of-3-days PV forecast; learned hold at the conformal level | 11/18 |
| 2 | 1d2ef25 | Median forecast; hold at the point-forecast level | 11/18 |
| 3 | f629fcf | Surplus = reached the level in force; refine at once when wakes are cheap; lazy buffer when a night does not fit | 11/18 |
| 4 | 7b8cd71 | Reservoir hold (sequent-peak level, re-bulk release); PV-overflow spending | 14/18 |
| 5 | 5a53f2f | When the hold may engage: `full` (iteration 4), `rule`, `rise` | 29, 30, **31**/32 |

Iterations 1–4 were scored on the paper grid (ratios 0.5, 1, 2; 18 cells
with ratio ≥ 1). Iteration 5 was scored on this document's grid layout, with
the baselines added at ratios 0.75, 1.5 and 3. The variant with the most
near-best cells (`rise`) was frozen as `sunsched3`; a unit test checks that
the frozen policy reproduces it step for step. The pass rules below were
drafted while iteration 3 was evaluated and were not changed afterwards. The
scorer's hypothesis labels and its Holm step were corrected to match this
text before iteration 5 was scored. Five design iterations on one corpus make
the development numbers optimistic, which is why the test uses two corpora
that no iteration has seen.

## Experimental grid (each test corpus)

Replicates are evaluation camera × weather year (PVGIS 2011–2015). One
simulator, the paper's (single pass from 80% initial charge).

- **Battery autonomy**: 0.5, 1, 2, 3, 5, 10, 30, 100 days.
- **Harvest-to-demand ratio**: 0.75, 1.0, 1.5, 2.0, 3.0.
- **Policies**: registered baselines `always_now` (Immediate), `ee_now`,
  `ceiling_now`, `lazy_defer`; further comparators `batch_anytime`,
  `defer_ceiling`, `ceiling_conformal`, `defer_ceiling_conformal`; the
  candidate `sunsched3`; and, exploratory only, its ablations
  `sunsched3_no_coalesce`, `sunsched3_no_pv`, `sunsched3_no_hold`.
- **Wake sweep**: wake energy 1, 2, 3, 6 J (default 12 J) at autonomy
  {0.5, 1, 2, 5, 30} × ratio {1, 2}, policies `always_now`, `lazy_defer`,
  `sunsched3`.

## Statistics

The camera is the unit. For a family of cells, each camera's paired effect is
averaged over the family's cells and the five weather years, giving one
number per camera. The confirmatory test is a **one-sided exact Wilcoxon
signed-rank test** on these camera means. HV2 and HV3 are Holm-corrected
together within a corpus (two tests). Intervals are 95% two-way bootstrap
intervals over cameras and years (4,000 draws). Each hypothesis is decided
separately on each corpus.

## Hypotheses

Immediate = `always_now`. Value = mean value score; life = battery years to
80% capacity (Xu et al. 2018 model).

**HV1: near-best in every regime.** For each cell with ratio ≥ 1 (32 cells),
let *V* be the best mean value among the registered baselines and *L* the
longest mean life among registered baselines with value ≥ *V* − 0.01. v3 is
near-best if its value ≥ *V* − 0.01 and its life ≥ 0.95 *L*. **Holds if v3 is
near-best in at least 75% of the 32 cells.** Descriptive (means), no p-value.

**HV2: v3 beats Immediate at small storage.** Cells autonomy {0.5, 1, 2} ×
ratio {1, 1.5, 2}. **Holds if** the pooled mean value gain over Immediate is
≥ 0.02, the one-sided camera-level test gives Holm p < 0.05, and the mean gain
is positive in at least 7 of the 9 cells.

**HV3: v3 lengthens battery life at large storage with surplus sun, at no
value cost.** Cells autonomy {30, 100} × ratio {1.5, 2, 3}. **Holds if** the
geometric-mean life ratio to Immediate is ≥ 1.15, the one-sided camera-level
test on log life gives Holm p < 0.05, the ratio is ≥ 1.15 in at least 5 of the
6 cells, and the pooled mean value change is ≥ −0.01.

**HV4: v3 never costs value against Immediate.** Over all 40 cells
(ratio 0.75 included), v3's mean value ≥ Immediate's − 0.01 in at least 90% of
cells. Descriptive.

**HV5: v3 does not lose when wakes are cheap.** In the wake sweep at 1 J and
3 J (20 cells), v3's mean value ≥ Immediate's − 0.01 in at least 75% of cells.
Descriptive.

**Primary claim:** HV1, HV2 and HV3 hold on a corpus. The claim generalises
if they hold on both. HV4 and HV5 are secondary.

### Development values (CCT20, frozen v3)

From `data/kaggle/v3dev5` (paper repository), scored with
`analysis/score_v2.py`; 9 evaluation cameras × 5 weather years per cell.
With 9 cameras the smallest possible one-sided p is 1/512 = 0.002.

| | Rule | CCT20 (development) |
|---|---|---|
| HV1 | near-best in ≥ 75% of 32 cells | 31/32 (97%); miss: 100 d / ratio 1, value −0.015 (life 1.23×) |
| HV2 | gain ≥ 0.02, Holm p < 0.05, ≥ 7/9 cells positive | +0.029 [0.013, 0.047], p_Holm = 0.004, 9/9 cells, 9/9 cameras |
| HV3 | life ≥ 1.15×, Holm p < 0.05, ≥ 5/6 cells, value ≥ −0.01 | 1.94× [1.91, 1.95], p_Holm = 0.004, 6/6 cells, value −0.0006 |
| HV4 | ≥ 90% of 40 cells within 0.01 of Immediate | 39/40 (97.5%) |
| HV5 | ≥ 75% of 20 wake cells within 0.01 of Immediate | 17/20 (85%) |

Near-best share of every policy under the HV1 rule (development): v3 0.969;
`defer_ceiling_conformal` 0.844; `lazy_defer` 0.812; `batch_anytime` and
`defer_ceiling` 0.750; Immediate 0.438; `ceiling_now`, `ceiling_conformal`
0.156; `ee_now` 0.125. Mean value over the 40 cells: v3 0.286, `lazy_defer`
0.282, `defer_ceiling_conformal` 0.277, Immediate 0.270.

## Descriptive measurements, not hypotheses

- Share of animal captures with the sun below the horizon, per corpus. From
  CCT20 (55.7%) and S10 (9.9%) we expect test A above 40% and test B below
  40%. Not a pass/fail item: v3's design does not depend on it.
- Near-best share (HV1 rule) of every policy, to show which fixed rule, if
  any, adapts as well as v3.
- Ablations of v3 on the HV1–HV3 cells.

## What each outcome means

| HV1 | HV2 | HV3 | Claim that can be made on that corpus |
|---|---|---|---|
| ✓ | ✓ | ✓ | One untuned controller matches the best fixed rule in each storage regime and beats Immediate at both ends. |
| ✓ | ✓ | ✗ | v3 adapts and gains value at small storage; its life gain does not transfer. |
| ✓ | ✗ | ✓ | v3 adapts and gains life at large storage; its small-storage value gain does not transfer. |
| ✗ | – | – | v3 is not near-best across regimes on this corpus; report which fixed rule is. |

## Known limitations, stated in advance

- Simulation only; node power figures are datasheet-typical and swept in the
  paper's sensitivity analysis. No hardware.
- One coordinate per corpus for the sun (region-level), PVGIS weather years.
- Captures from several calendar years at one camera are laid onto one year.
- The frozen MobileNetV3 trunk is weak on unseen cameras; the hypotheses
  concern differences between policies that share the classifier.
- v3 was designed after the v1 results were known, on CCT20, which shares a
  region and a weather site with test A. Test B guards against a
  region-specific design.
- Each simulation is one weather year started on 1 January at 80% charge.
  The hold learns its level from that year only; a site whose worst season
  comes after a sunny start (test B is near the equator) is a harder case
  than the development data.
