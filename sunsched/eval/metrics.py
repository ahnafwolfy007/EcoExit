"""Metrics for one simulated deployment.

Primary task metric is the value score: value-weighted fraction of frames
that end with a correct label, where a frame that was missed, left
unclassified, or labelled wrong scores zero. Raw accuracy is reported too, but
it is dominated by whichever class is most common at a camera.
"""
from typing import Dict

import numpy as np

from sunsched.env.battery import aging_summary
from sunsched.sim.simulator import (DROPPED, REFINED_LATER, REFINED_NOW, TRIAGE,
                                    UNCLASSIFIED, SimResult)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return float(np.mean(f1s)) if f1s else float("nan")


def reference_scores(frames: np.ndarray, outcomes) -> Dict[str, float]:
    """What the classifier alone allows at this camera, ignoring energy:
    every frame triaged only, and every frame refined with the full model."""
    labels = outcomes.labels[frames]
    values = outcomes.values[frames]
    total = max(values.sum(), 1e-12)
    out = {}
    for op in outcomes.pred:
        out[f"ref_value_{op}"] = float((values * (outcomes.pred[op][frames] == labels)).sum() / total)
    return out


def classification_metrics(res: SimResult, outcomes, slot_minutes: int) -> Dict[str, float]:
    frames = res.frames
    labels = outcomes.labels[frames]
    values = outcomes.values[frames]
    pred = res.final_label
    correct = (pred == labels) & (pred >= 0)
    empty = outcomes.empty_idx
    animal_true = labels != empty
    animal_pred = (pred >= 0) & (pred != empty)

    n = max(len(frames), 1)
    lat_h = np.where(res.final_slot >= 0, res.final_slot - res.capture_slot, 0) * slot_minutes / 60.0
    later = res.route == REFINED_LATER
    ref = reference_scores(frames, outcomes)
    value_score = float((values * correct).sum() / max(values.sum(), 1e-12))
    lo, hi = ref["ref_value_triage"], ref.get("ref_value_full", np.nan)

    return dict(
        n_frames=int(len(frames)),
        value_score=value_score,
        # Share of the achievable improvement over triage-only that the policy
        # actually realised; 1.0 would be "every frame refined for free".
        gain_captured=float((value_score - lo) / (hi - lo)) if hi - lo > 1e-9 else float("nan"),
        accuracy=float(correct.mean()) if len(frames) else float("nan"),
        macro_f1=macro_f1(labels, pred) if len(frames) else float("nan"),
        animal_recall=float(animal_pred[animal_true].mean()) if animal_true.any() else float("nan"),
        empty_false_alarm=float(animal_pred[~animal_true].mean()) if (~animal_true).any() else float("nan"),
        dropped_frac=float(np.mean(res.route == DROPPED)) if len(frames) else 0.0,
        unclassified_frac=float(np.mean(res.route == UNCLASSIFIED)) if len(frames) else 0.0,
        triage_final_frac=float(np.mean(res.route == TRIAGE)) if len(frames) else 0.0,
        refined_now_frac=float(np.mean(res.route == REFINED_NOW)) if len(frames) else 0.0,
        refined_later_frac=float(np.mean(later)) if len(frames) else 0.0,
        latency_mean_h=float(lat_h.mean()) if len(frames) else 0.0,
        latency_p95_h=float(np.percentile(lat_h, 95)) if len(frames) else 0.0,
        latency_refined_later_p95_h=float(np.percentile(lat_h[later], 95)) if later.any() else 0.0,
        **ref,
    )


def energy_battery_metrics(res: SimResult, cfg, days: float) -> Dict[str, float]:
    soc = res.soc
    ageing = aging_summary(soc, res.temp_c, cfg.slot_seconds, days, cfg.aging, cfg.battery.capacity_wh)
    return dict(
        soc_mean=float(soc.mean()),
        soc_p95=float(np.percentile(soc, 95)),
        frac_time_soc_above_90=float(np.mean(soc > 0.9)),
        battery_temp_mean_c=float(res.temp_c.mean()),
        ceiling_mean=float(res.ceiling.mean()),
        harvested_kj=res.harvested_j / 1e3,
        consumed_kj=res.consumed_j / 1e3,
        curtailed_kj=res.curtailed_j / 1e3,
        b_wakes_per_day=res.b_wakes / max(days, 1e-9),
        dead_frac=res.dead_slots / max(len(soc), 1),
        deferred=int(res.deferred),
        reserve_coverage=(float("nan") if res.reserve_coverage is None else res.reserve_coverage),
        **ageing,
    )


def summarize(res: SimResult, outcomes, cfg, days: float) -> Dict[str, float]:
    return {**classification_metrics(res, outcomes, cfg.solar.slot_minutes),
            **energy_battery_metrics(res, cfg, days)}
