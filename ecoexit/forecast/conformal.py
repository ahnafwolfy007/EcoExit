"""Solar harvest forecasting with a distribution-free risk bound.

Two layers, matching Contribution B in the report:

  1. `ClearSkyPersistence` -- a deliberately cheap point forecaster. It
     predicts tomorrow's clear-sky index from a weighted average of recent
     clear-sky indices (removing the deterministic diurnal/seasonal cycle
     first, which is exactly what a naive "look at last night's irradiance"
     baseline cannot do).

  2. `AdaptiveConformalForecaster` -- wraps *any* point forecaster and turns
     its residuals into a calibrated one-sided lower bound on cumulative
     harvested energy over a horizon, using Adaptive Conformal Inference
     (Gibbs & Candes, 2021): the miscoverage target alpha_t is nudged after
     every slot based on whether the previous bound actually held, so
     coverage self-corrects across seasons/sites without assuming the solar
     process is Gaussian or stationary.

This is the piece that has no equivalent anywhere in the 40-paper corpus:
HarvSched and Bullo et al. condition on point estimates or learned state,
never on a bound with an explicit coverage guarantee.
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np


class ClearSkyPersistence:
    """kc_hat[t+h] = weighted average of kc over the last `window` slots at
    the *same time of day*, i.e. persistence of the clear-sky index, not of
    raw irradiance (irradiance persistence would just be predicting sunrise)."""

    def __init__(self, slots_per_day: int, window_days: int = 3, decay: float = 0.5):
        self.spd = slots_per_day
        self.window_days = window_days
        self.decay = decay

    def predict(self, kc_history: np.ndarray, horizon_slots: int) -> np.ndarray:
        """Given all observed kc up to now, predict kc for the next
        `horizon_slots`, reusing same-time-of-day history with exponential
        recency weighting across days."""
        n = len(kc_history)
        preds = np.empty(horizon_slots)
        for h in range(horizon_slots):
            future_slot = n + h
            vals, wts = [], []
            for d in range(1, self.window_days + 1):
                idx = future_slot - d * self.spd
                if 0 <= idx < n:
                    vals.append(kc_history[idx])
                    wts.append(self.decay ** (d - 1))
            if vals:
                preds[h] = np.average(vals, weights=wts)
            else:
                preds[h] = kc_history[-min(self.spd, n):].mean() if n else 0.5
        return np.clip(preds, 0.0, 1.0)


@dataclass
class ACIState:
    alpha_t: float
    scores: List[float] = field(default_factory=list)
    coverage_log: List[int] = field(default_factory=list)


class AdaptiveConformalForecaster:
    """Adaptive Conformal Inference (Gibbs & Candes 2021) applied to the
    *horizon-minimum cumulative harvest*, so one calibrated quantity covers
    the whole planning window instead of one bound per slot (which would
    need a union bound and blow up alpha).

    score_t = point_forecast_of_horizon_harvest - actual_horizon_harvest
    A one-sided lower bound is: harvest_hat - Quantile_{1-alpha}(scores)
    """

    def __init__(self, alpha: float, gamma: float, window: int = 200):
        self.alpha0 = alpha
        self.gamma = gamma
        self.window = window
        self.state = ACIState(alpha_t=alpha)

    def _quantile(self) -> float:
        if not self.state.scores:
            return 0.0
        recent = self.state.scores[-self.window:]
        q = np.clip(1.0 - self.state.alpha_t, 0.01, 0.99)
        return float(np.quantile(recent, q))

    def lower_bound(self, point_forecast_j: float) -> float:
        return max(0.0, point_forecast_j - self._quantile())

    def update(self, point_forecast_j: float, actual_j: float, prior_bound: float):
        """Call once actual harvest for the scored horizon is observed."""
        score = point_forecast_j - actual_j
        self.state.scores.append(score)
        covered = int(actual_j >= prior_bound)
        self.state.coverage_log.append(covered)
        # ACI update: push alpha_t up (bound gets looser) after a miss,
        # down (tighter) after a hit -- self-correcting, distribution-free.
        err_t = 1 - covered
        self.state.alpha_t = float(np.clip(
            self.state.alpha_t + self.gamma * (self.alpha0 - err_t), 0.01, 0.5
        ))

    def empirical_coverage(self, window: int = None) -> float:
        log = self.state.coverage_log if window is None else self.state.coverage_log[-window:]
        return float(np.mean(log)) if log else float("nan")


def simulate_forecast_and_calibration(kc_true: np.ndarray, slots_per_day: int,
                                        horizon_slots: int, ghi_clear: np.ndarray,
                                        panel_area_m2: float, panel_eff: float,
                                        slot_seconds: int, alpha: float, gamma: float,
                                        forecaster: "ClearSkyPersistence" = None):
    """Roll a persistence forecaster + ACI calibrator forward over a whole
    trace, scoring every `horizon_slots`-block. Returns arrays aligned to
    block index: point-forecast harvest, actual harvest, calibrated lower
    bound, and the running empirical coverage."""
    forecaster = forecaster or ClearSkyPersistence(slots_per_day)
    aci = AdaptiveConformalForecaster(alpha, gamma)

    n = len(kc_true)
    n_blocks = n // horizon_slots
    point_j = np.zeros(n_blocks)
    actual_j = np.zeros(n_blocks)
    lower_j = np.zeros(n_blocks)
    alpha_t_trace = np.zeros(n_blocks)

    for b in range(n_blocks):
        start = b * horizon_slots
        end = start + horizon_slots
        history = kc_true[:start]
        if len(history) < slots_per_day:
            kc_hat = np.full(horizon_slots, kc_true[:max(start, 1)].mean() if start else 0.5)
        else:
            kc_hat = forecaster.predict(history, horizon_slots)

        ghi_hat = ghi_clear[start:end] * kc_hat
        p_hat_w = ghi_hat * panel_area_m2 * panel_eff
        forecast_j = float((p_hat_w * slot_seconds).sum())

        ghi_actual = ghi_clear[start:end] * kc_true[start:end]
        p_actual_w = ghi_actual * panel_area_m2 * panel_eff
        real_j = float((p_actual_w * slot_seconds).sum())

        bound = aci.lower_bound(forecast_j)
        point_j[b] = forecast_j
        actual_j[b] = real_j
        lower_j[b] = bound
        alpha_t_trace[b] = aci.state.alpha_t

        aci.update(forecast_j, real_j, bound)

    return dict(point_j=point_j, actual_j=actual_j, lower_j=lower_j,
                alpha_t=alpha_t_trace, aci=aci)
