# Pre-registration v3: when does deferred inference help?

Written and committed **before any image of the test corpora below was
downloaded**. The predictor, the controller, its constants, the job grid and
the pass rules are fixed here. `analysis/score_v3.py`, committed with this
file, implements them exactly (`verdicts`). Anything reported beyond them is
labelled exploratory. The verdicts of v1 (`PREREGISTRATION.md`) and v2
(`PREREGISTRATION_v2.md`) stand unchanged: v2's primary claim was not met.

## What was seen before this file

- **Images and results:** CCT20, `cct_new`, Serengeti S10 and S11. All four are
  development data from here on.
- **Test corpora, metadata only:** LILA's per-image table was read by
  `pipeline/00b_verify_eligibility.py` (counts, time stamps, shares of empty
  frames, number of label categories, reachable URLs) and
  `pipeline/00c_probe_public.py` plus one label-wise HEAD probe (which rows
  point to files that are not public). No image was opened, no classifier was
  run, no capture stream was simulated and `W` was not computed.
- **A pipeline check** (`qsmoke`) runs the test code on NZ trail-camera images,
  a corpus that failed the eligibility gates and is not a test corpus.

## Test corpora

| Corpus | Region | Weather (PVGIS 5.3) | Eligible cameras |
|---|---|---|---|
| `lila_swg` | Annamites, Laos and Vietnam | 16.20 N, 107.60 E | 871 |
| `lila_orinoquia` | Meta, Colombia | 4.15 N, 73.63 W | 47 |
| `lila_nkhotakota` | Nkhotakota, Malawi | 12.90 S, 34.10 E | 62 |
| `lila_wellington` | Wellington, New Zealand | 41.29 S, 174.78 E | 179 |
| `lila_lynx` | Washington State, USA | 48.60 N, 119.50 W | 415 |

Selection is code (`sunsched/data/lila.py`): 25 evaluation cameras, 5
calibration cameras, the rest for training, in SHA-256 order balanced over
activity. At least 200 frames and 60 days per camera; the 365-day window with
most captures; at most 5,000 frames in time order.

**Changes from the plan (PLAN_V2), all made on metadata alone.**
1. AMMonitor (northern Appalachians) is replaced by WSU Lynx: 33% of its
   sampled files are not public. The rule, stated before the probe, was to
   keep a corpus only if at least 99% of its images of anything but people can
   be downloaded.
2. Images of people and vehicles are dropped in every corpus. LILA withholds
   them in three of the five (Annamites, Orinoquía, Nkhotakota); the other two
   contain none. A real node would see them.
3. For the Annamites the literal 99% rule fails (88.5% of images without a
   person label are reachable). In a sample of 1,500 rows every missing file
   lies in a folder the publisher calls `private`. Of 62 labels with at least
   300 rows, two, `ignore` and `human`, are missing entirely, and one rare
   label is missing 10% of its files; all others are complete within the
   sample. The loader therefore drops every row labelled `ignore` or a person
   label, and every row whose file lies in that folder, and the corpus is kept.
   The literal rule is broken here and this is the reason why.
4. The classifier hypothesis of PLAN_V2 (its H2) is not tested here: it needs a GPU
   on every corpus and the weekly quota does not allow it. It is reported on
   development corpora only and no test claim is made.
5. Three weather years (2019, 2021, 2023) instead of five, for run time: 25
   cameras x 19 cells x 3 years x 15 policies = 21,375 runs per corpus, plus
   1,800 runs of other nodes.

If more than 1% of a corpus's selected images fail to download, the corpus is
still scored and the share is reported. If a corpus ends with fewer than 10
evaluation cameras with results, it is excluded and the rules below use the
remaining n corpora. A crash before any simulated row may be fixed and re-run;
a re-run after rows exist is reported as such.

## Classifier, node and grid

A camera's capture stream is its window of at most 365 days. Where the window is
shorter than a year the rest of the simulated year has no captures (Washington:
cameras ran in June to October only).

The frozen-trunk MobileNetV3-Large pipeline of the development study, with
heads trained per corpus on its training cameras (stage 1 to 3, CPU). Node
constants, ageing model and value weights are unchanged. Grid: autonomy 0.5,
1, 2, 5, 10 d × ratio 1, 1.5, 2 and 30, 100 d × ratio 1.5, 2 (19 cells), one
unscored warm-up year before the scored year.

## Policies

- `always_now` (Immediate), `lazy_defer`, `lazy_c85 … lazy_c40` (lazy deferral
  with a fixed charge ceiling), `ceiling_now` (predicted-harvest ceiling),
  `sunsched3` (the frozen controller of v2).
- **Registered controller `v7`** and its front `v7, v7_c85, v7_c70, v7_c60, v7_c50, v7_c40`
  (`sunsched/policies/v5.py`, constants in `sunsched/experiment.py`).
  Uncertain frames are queued and refined by late paid wakes and by free wakes
  from energy the battery would refuse; confident frames only from refused
  energy; the sunset reserve is the night's sleep energy, at most lazy
  deferral's buffer; where that energy does not fit the battery, the rule of
  lazy deferral applies; a reservoir hold lowers the ceiling after three
  surplus days, unless harvest is under 1.3 times demand; a cap
  c in {1, 0.85, 0.7, 0.6, 0.5, 0.4} always applies. Constants unchanged from the code.

## Hypotheses

Unit: camera (effects averaged over cells and years). Small storage = 0.5, 1,
2 d; large = 30, 100 d. n = number of corpora.

- **H1 (primary), predictor.** (a) Pooled Spearman between `W` and the
  small-storage value gain of `lazy_defer` over Immediate ≥ 0.5, one-sided
  p < 0.001. (b) Positive inside at least n − 1 corpora. (c) Cameras with
  W ≥ 0.05 gain at least three times as much as cameras with W < 0.02; both
  groups need ≥ 10 cameras, otherwise (c) counts as not met.
- **H1n (secondary), nodes.** The same test across nodes: wake energy 3 J and
  35 J, sleep power 0.5 mW, and the default (`qnode` runs, small storage, one
  weather year), with `W` recomputed for each. Over all camera-node pairs,
  Spearman(W, gain) >= 0.5 and above that of the frame count; within a camera,
  across nodes, median Spearman >= 0.5 with one-sided Wilcoxon p < 0.001.
  A 12 h deadline is also run and reported, but not tested: daylight-only
  refinement loses night frames there, which an energy bound cannot see.
- **H2, deferral where it is predicted.** On cameras with W ≥ 0.05 the
  registered controller gains ≥ 0.03 over Immediate at small storage, one-sided
  exact Wilcoxon p < 0.001.
- **H3, life.** At large storage: battery life ≥ 1.5 × Immediate (p < 0.05,
  Holm over corpora) with a value change ≥ −0.01, on at least n − 1 corpora.
- **H4 and H4L, front.** Two rules: PLAN_V2's rule was fixed early, and the development
  data say where the gain lies.
  - **H4 (as in PLAN_V2).** Hypervolume of the registered front over the whole
    grid at least 3% above lazy deferral with static ceilings (six levels each),
    p < 0.05 (Holm over corpora), on at least 3 corpora (2 if n < 5). On
    development it holds on S10, S11 and cct_new (+3.3%, +5.1% and +4.0% on the test
    cells) but not on CCT20 (+2.6%), so it may fail.
  - **H4L (added before the freeze).** The same at large storage (30 and 100 d):
    at least 5% above, p < 0.05, on at least n - 1 corpora.

H1 is the claim of the paper. H1n, H2, H3, H4 and H4L are secondary. All are reported
whatever happens, with the pooled Spearman of the frame count beside H1.

## Development evidence behind the choices

Development corpora (54 cameras, frozen-trunk outcomes; 2011 and 2013 weather,
365-day warm-up). Policies were chosen on these numbers; they are not
confirmatory. v7 is the fourth look: v5 (back-off and late wakes), v5 with a
charge cap, v6 (a buffer rule for the sunset reserve), v7 (the fallback to lazy
deferral where a night does not fit). The front counts six levels on both sides.

| | CCT20 (9) | cct_new (29) | S10 (6) | S11 (10) |
|---|---|---|---|---|
| Median W | 0.067 | 0.003 | 0.134 | 0.119 |
| Spearman(W, gain), small storage | 0.45 | 0.36 | 0.49 | 0.37 |
| Gain of lazy deferral, small storage | +0.026 | +0.019 | +0.144 | +0.086 |
| Gain of v7, small storage | +0.026 | +0.019 | +0.138 | +0.092 |
| v7 life at large storage (Immediate = 1) | 2.06 | 2.06 | 1.95 | 2.00 |
| v7 value change at large storage | -0.000 | -0.001 | +0.030 | -0.000 |
| v7 front against lazy + ceilings, whole grid (21 cells) | +2.6% | +5.0% | +5.5% | +6.4% |
| same, 19 test cells | +2.6% | +4.0% | +3.3% | +5.1% |
| same, large storage | +10.8% | +10.5% | +10.3% | +12.8% |
| same, small storage | -0.3% | -0.0% | -0.8% | +0.5% |

- Pooled over 54 cameras: Spearman(W, gain) 0.588 (one-sided p = 1e-6), frame
  count 0.589. Cameras with W >= 0.05 gain 0.096, with W < 0.02 gain 0.019. At one
  node W is no better than the frame count; that is what H1n is for.
- Nodes (wake 3 and 35 J, sleep 0.5 mW, default; 216 camera-node pairs):
  Spearman(W, gain) 0.568, frame count 0.494; within a camera, median 0.80 over
  51 cameras, positive in 78%, p = 1e-7. Over all nodes including the 12 h
  deadline 0.588 and 0.501.
- The margin of 0.1 between W and the frame count in an earlier draft of H1n was
  dropped after these numbers (0.07 and 0.09): "above" is the test, and the
  within-camera test is the stricter one. This change was made before the
  freeze, on development data only.
- The gate of PLAN_V2 (D6) is met: the front is at least 3% above on S10 and is
  not below on the others.
- The classifier gates D2, D3 and D4 of PLAN_V2 fail on the development corpora
  (memory, triage memory and the learned gain head). D1 selects MobileNetV4
  Conv-Medium. None of it enters the simulations of this registration.

