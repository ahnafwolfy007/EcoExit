# SunSched results

Generated 2026-09-28 14:59. Irradiance source: **pvgis**. Preset: **full**. Site: cct_region. Weather years: [2011, 2012, 2013, 2014, 2015].

## 1. Why schedule in time

Across 53,850 animal captures from all 20 CCT20 cameras, **55.7%** arrive with the sun below the horizon, when harvest is zero. Only **17.9%** arrive with the sun above 40 degrees, which delivers **54.4%** of the energy. Hourly correlation between the two: -0.17.

![diel mismatch](figures/diel_mismatch.png)

Energy per action (J): `{'sleep_per_slot': 0.6, 'capture_day': 0.4, 'capture_night': 1.2, 'triage': 0.05102482944, 'store': 0.02, 'tier_b_wake': 12.0, 'refine_lite': 0.375572672, 'refine_full': 0.507017312}`

## 2. Classifier (frozen MobileNetV3-Large, trained exit heads)

Evaluated on the 9 held-out cameras (trans_test); temperatures fitted on trans_val.

| input height | exit | MMACs | accuracy | macro-F1 | animal-vs-empty | ECE before -> after |
|---|---|---|---|---|---|---|
| 160 | 0 | 51.2 | 0.227 | 0.170 | 0.833 | 0.166 -> 0.107 |
| 160 | 1 | 108.9 | 0.222 | 0.193 | 0.849 | 0.243 -> 0.007 |
| 160 | 2 | 153.8 | 0.219 | 0.215 | 0.882 | 0.218 -> 0.019 |
| 320 | 0 | 204.7 | 0.271 | 0.193 | 0.845 | 0.390 -> 0.022 |
| 320 | 1 | 434.1 | 0.234 | 0.190 | 0.860 | 0.273 -> 0.022 |
| 320 | 2 | 609.4 | 0.327 | 0.302 | 0.903 | 0.145 -> 0.120 |

## 3. Main comparison

Mean over held-out cameras x weather years, with 95% bootstrap CIs.

**Harvest-to-demand ratio 0.5**

| policy | value score | animal recall | missed | latency p95 (h) | battery years | time SoC>90% | wakes/day |
|---|---|---|---|---|---|---|---|
| always_now | 0.198 [0.173, 0.223] | 0.597 | 0.376 | 0.0 | 17.57 [17.21, 17.93] | 0.0% | 1.09 |
| sunsched_no_ceiling | 0.195 [0.175, 0.217] | 0.650 | 0.331 | 0.8 | 17.22 [16.85, 17.57] | 0.0% | 0.75 |
| lazy_defer | 0.185 [0.159, 0.213] | 0.653 | 0.273 | 48.0 | 16.65 [16.28, 17.00] | 0.0% | 0.14 |
| sunsched | 0.175 [0.156, 0.195] | 0.592 | 0.391 | 0.7 | 17.94 [17.58, 18.31] | 0.0% | 0.72 |
| sunsched_no_conformal | 0.175 [0.156, 0.194] | 0.592 | 0.391 | 0.6 | 17.94 [17.59, 18.29] | 0.0% | 0.73 |
| triage_only | 0.170 [0.137, 0.204] | 0.666 | 0.255 | 0.0 | 16.35 [16.00, 16.70] | 0.0% | 0.00 |
| sunsched_no_defer | 0.168 [0.150, 0.187] | 0.571 | 0.392 | 0.0 | 17.95 [17.59, 18.31] | 0.0% | 0.75 |
| ceiling_now | 0.167 [0.149, 0.187] | 0.537 | 0.412 | 0.0 | 18.11 [17.75, 18.49] | 0.0% | 0.87 |
| ee_now | 0.156 [0.140, 0.173] | 0.608 | 0.316 | 0.0 | 17.28 [16.95, 17.64] | 0.0% | 0.62 |
| sunsched_lite | 0.146 [0.121, 0.172] | 0.600 | 0.329 | 1.0 | 17.41 [17.06, 17.74] | 0.0% | 0.06 |

**Harvest-to-demand ratio 1.0**

| policy | value score | animal recall | missed | latency p95 (h) | battery years | time SoC>90% | wakes/day |
|---|---|---|---|---|---|---|---|
| always_now | 0.344 [0.290, 0.401] | 0.955 | 0.000 | 0.0 | 11.77 [11.50, 12.03] | 0.0% | 1.95 |
| sunsched_no_ceiling | 0.315 [0.273, 0.362] | 0.968 | 0.000 | 0.9 | 11.11 [10.83, 11.38] | 0.0% | 1.24 |
| lazy_defer | 0.312 [0.258, 0.370] | 0.941 | 0.000 | 43.9 | 11.05 [10.79, 11.31] | 0.0% | 0.89 |
| sunsched_no_conformal | 0.311 [0.269, 0.357] | 0.954 | 0.015 | 0.9 | 15.38 [15.07, 15.67] | 0.0% | 1.30 |
| sunsched | 0.311 [0.266, 0.358] | 0.955 | 0.014 | 0.9 | 15.38 [15.09, 15.67] | 0.0% | 1.30 |
| sunsched_no_defer | 0.297 [0.257, 0.344] | 0.920 | 0.018 | 0.0 | 15.40 [15.10, 15.69] | 0.0% | 1.37 |
| ceiling_now | 0.296 [0.253, 0.345] | 0.922 | 0.020 | 0.0 | 15.76 [15.46, 16.07] | 0.0% | 1.67 |
| ee_now | 0.238 [0.204, 0.274] | 0.911 | 0.000 | 0.0 | 11.73 [11.45, 11.99] | 0.0% | 1.87 |
| triage_only | 0.224 [0.183, 0.272] | 0.899 | 0.000 | 0.0 | 10.09 [9.75, 10.44] | 7.6% | 0.00 |
| sunsched_lite | 0.223 [0.181, 0.267] | 0.896 | 0.001 | 11.7 | 14.85 [14.56, 15.12] | 0.0% | 0.08 |

**Harvest-to-demand ratio 2.0**

| policy | value score | animal recall | missed | latency p95 (h) | battery years | time SoC>90% | wakes/day |
|---|---|---|---|---|---|---|---|
| always_now | 0.344 [0.292, 0.402] | 0.955 | 0.000 | 0.0 | 7.71 [7.57, 7.85] | 90.2% | 1.95 |
| lazy_defer | 0.343 [0.293, 0.402] | 0.955 | 0.000 | 10.7 | 7.70 [7.57, 7.84] | 91.0% | 1.19 |
| ceiling_now | 0.343 [0.291, 0.400] | 0.955 | 0.000 | 0.0 | 14.58 [14.28, 14.85] | 0.0% | 1.95 |
| ee_now | 0.343 [0.292, 0.401] | 0.955 | 0.000 | 0.0 | 7.71 [7.57, 7.85] | 90.2% | 1.95 |
| sunsched | 0.315 [0.273, 0.364] | 0.968 | 0.000 | 0.8 | 14.51 [14.25, 14.80] | 0.0% | 1.35 |
| sunsched_no_ceiling | 0.315 [0.271, 0.364] | 0.968 | 0.000 | 0.9 | 7.70 [7.57, 7.84] | 91.0% | 1.28 |
| sunsched_no_conformal | 0.315 [0.270, 0.364] | 0.968 | 0.000 | 0.8 | 14.51 [14.24, 14.80] | 0.0% | 1.35 |
| sunsched_no_defer | 0.304 [0.260, 0.355] | 0.938 | 0.000 | 0.0 | 14.51 [14.21, 14.80] | 0.0% | 1.45 |
| triage_only | 0.224 [0.184, 0.271] | 0.899 | 0.000 | 0.0 | 7.70 [7.55, 7.83] | 92.1% | 0.00 |
| sunsched_lite | 0.223 [0.185, 0.270] | 0.897 | 0.000 | 11.6 | 14.32 [14.03, 14.60] | 0.0% | 0.08 |

![routes](figures/routes.png)

## 4. Kill test

**FAIL** against ee_now, ceiling_now, lazy_defer (margins: {'value_margin': 0.02, 'life_ratio': 1.15, 'value_tolerance': 0.01, 'life_tolerance': 0.05}). Per regime: {'0.5': False, '1.0': False, '2.0': False}.

| ratio | versus | value diff | p (Holm) | life ratio | p (Holm) | beats |
|---|---|---|---|---|---|---|
| 0.5 | ee_now | +0.019 | 0.00312 | x1.04 | 3.41e-13 | False |
| 0.5 | ceiling_now | +0.008 | 0.467 | x0.99 | 3.41e-13 | False |
| 0.5 | lazy_defer | -0.010 | 0.467 | x1.08 | 3.41e-13 | False |
| 1.0 | ee_now | +0.073 | 9.08e-05 | x1.31 | 3.41e-13 | True |
| 1.0 | ceiling_now | +0.014 | 0.604 | x0.98 | 3.41e-13 | False |
| 1.0 | lazy_defer | -0.002 | 0.95 | x1.39 | 3.41e-13 | True |
| 2.0 | ee_now | -0.028 | 0.102 | x1.88 | 3.41e-13 | False |
| 2.0 | ceiling_now | -0.028 | 0.102 | x1.00 | 3.41e-13 | False |
| 2.0 | lazy_defer | -0.028 | 0.102 | x1.88 | 3.41e-13 | False |

## 5. Accuracy vs battery-life frontiers

![frontier 0.5](figures/frontier_0.5.png)
![frontier 1.0](figures/frontier_1.0.png)
![frontier 2.0](figures/frontier_2.0.png)

## 6. Regime: where scheduling matters

![regime](figures/regime.png)

## 7. Sensitivity to assumed constants

SunSched minus the best of the strong baselines at each setting (value score; battery-life ratio).

| assumption | setting | value diff | life ratio |
|---|---|---|---|
| aging.k_T | high | -0.010 | x1.08 |
| aging.k_T | low | -0.010 | x1.07 |
| aging.k_sigma | high | -0.010 | x1.12 |
| aging.k_sigma | low | -0.010 | x1.04 |
| aging.k_t | high | -0.010 | x1.08 |
| aging.k_t | low | -0.010 | x1.08 |
| battery.capacity_wh | high | -0.013 | x0.99 |
| battery.capacity_wh | low | -0.015 | x1.03 |
| control.deadline_h | high | -0.010 | x1.08 |
| control.deadline_h | low | -0.009 | x1.08 |
| control.min_ceiling | high | -0.001 | x1.06 |
| control.min_ceiling | low | -0.019 | x1.09 |
| node.e_capture_night_j | high | -0.012 | x1.07 |
| node.e_capture_night_j | low | -0.010 | x1.08 |
| node.p_sleep_w | high | -0.001 | x1.00 |
| node.p_sleep_w | low | +0.006 | x0.97 |
| node.tier_a_pj_per_mac | high | -0.010 | x1.08 |
| node.tier_a_pj_per_mac | low | -0.010 | x1.08 |
| node.tier_b_active_w | high | -0.012 | x1.08 |
| node.tier_b_active_w | low | -0.009 | x1.08 |
| node.tier_b_wake_j | high | -0.055 | x1.14 |
| node.tier_b_wake_j | low | +0.001 | x1.00 |
| site | bergen | +0.005 | x1.11 |
| site | dhaka | -0.010 | x1.10 |
| site | munich | -0.017 | x1.10 |
| site | nairobi | -0.011 | x1.08 |
| site | phoenix | -0.008 | x1.10 |
| solar.enclosure_gain_c | high | -0.010 | x1.08 |
| solar.enclosure_gain_c | low | -0.010 | x1.08 |

## 7b. Battery autonomy: can scheduling matter at all?

Days of load the battery holds, swept as a first-class axis rather than fixed. `ceil@floor` is the fraction of slots where the conformal ceiling was clipped to `control.min_ceiling`: near 1, the risk-controlled reserve cannot steer the battery whatever its coverage, the cell ages by calendar, and the comparison measures that floor rather than the schedule.

This sweep is **exploratory, not pre-registered**. The kill test in section 4 stands as the pre-registered result at the default sizing. Every level is reported here precisely so that no single one is selected after the fact.

| days | ratio | sunsched value | vs ee_now | vs ceiling_now | vs lazy_defer | life x best | ceil@floor |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.5 | 0.066 | +0.001 | +0.001 | +0.000 | x1.00 | 0.00 |
| 0.5 | 1 | 0.173 | +0.005 | +0.005 | -0.068 | x1.08 | 0.00 |
| 0.5 | 2 | 0.195 | -0.007 | -0.006 | -0.090 | x1.09 | 0.00 |
| 1 | 0.5 | 0.054 | -0.000 | -0.000 | -0.000 | x1.00 | 0.00 |
| 1 | 1 | 0.199 | +0.004 | +0.012 | -0.069 | x1.08 | 0.00 |
| 1 | 2 | 0.222 | -0.037 | -0.020 | -0.109 | x1.20 | 0.00 |
| 2 | 0.5 | 0.033 | +0.000 | +0.000 | +0.000 | x1.00 | 0.00 |
| 2 | 1 | 0.222 | +0.016 | +0.036 | -0.036 | x1.31 | 0.01 |
| 2 | 2 | 0.296 | +0.005 | +0.045 | -0.043 | x1.56 | 0.03 |
| 3 | 0.5 | 0.018 | +0.000 | +0.000 | +0.000 | x1.00 | 0.00 |
| 3 | 1 | 0.228 | +0.017 | +0.046 | -0.027 | x1.47 | 0.15 |
| 3 | 2 | 0.300 | -0.014 | +0.043 | -0.040 | x1.84 | 0.38 |
| 5 | 0.5 | 0.008 | +0.001 | +0.001 | +0.000 | x1.00 | 0.23 |
| 5 | 1 | 0.228 | +0.013 | +0.043 | -0.023 | x1.57 | 0.53 |
| 5 | 2 | 0.305 | -0.024 | +0.031 | -0.037 | x1.99 | 0.80 |
| 10 | 0.5 | 0.014 | +0.003 | +0.003 | -0.000 | x1.00 | 0.97 |
| 10 | 1 | 0.229 | +0.015 | +0.033 | -0.025 | x1.52 | 0.97 |
| 10 | 2 | 0.312 | -0.026 | -0.005 | -0.031 | x2.02 | 0.97 |
| 30 | 0.5 | 0.029 | +0.006 | -0.002 | +0.002 | x1.00 | 0.98 |
| 30 | 1 | 0.231 | +0.007 | +0.026 | -0.055 | x1.54 | 0.98 |
| 30 | 2 | 0.315 | -0.027 | -0.023 | -0.029 | x2.00 | 0.98 |
| 100 | 0.5 | 0.099 | +0.012 | +0.005 | -0.020 | x1.04 | 0.98 |
| 100 | 1 | 0.291 | +0.060 | +0.022 | -0.014 | x1.46 | 0.98 |
| 100 | 2 | 0.315 | -0.028 | -0.027 | -0.028 | x1.94 | 0.98 |
| 175 | 0.5 | 0.178 | +0.019 | +0.013 | -0.008 | x1.08 | 0.98 |
| 175 | 1 | 0.309 | +0.072 | +0.017 | -0.002 | x1.39 | 0.98 |
| 175 | 2 | 0.315 | -0.028 | -0.028 | -0.028 | x1.88 | 0.98 |

## 8. What is assumed, not measured

- No hardware: node power figures are datasheet-typical assumptions (`NodeCfg`), swept in section 7.
- Battery ageing uses the Xu et al. (IEEE Trans. Smart Grid 2018) semi-empirical model; constants must be checked against the paper and are swept +/-50%.
- Enclosure heating (battery temperature above air temperature) is an assumption, swept 0-30 C.
- Camera coordinates are region-level (CCT does not publish them); capture clock times may include daylight saving time.
- Captures from several calendar years at one camera are laid onto one simulated year.
- MAC counts are measured from the real network; energy per MAC is assumed.
- `sunsched` and `sunsched_no_conformal` differ only in the reserve forecaster, so compare them on `reserve_coverage` in `tables/runs.csv`: the conformal bound should sit near 1 - alpha and the point forecast well below it. Meeting that target is not the same as steering the battery. If `ceiling_at_floor_frac` is near 1, the bound was clipped to `control.min_ceiling` before it reached the cell, and no lifetime difference may be credited to the reserve however good its coverage looks. That is what happens when the battery holds many days of load: see `days_of_autonomy` in `environment_summary.json`, and the `control.min_ceiling` and `battery.capacity_wh` rows of section 7.
