"""Risk-controlled night reserve via Adaptive Conformal Inference.

The quantity forecast is, for each sunset, the energy the node must already
hold to keep capturing and triaging every event over the following horizon:
the largest cumulative shortfall of essential load over harvest. It is a
scalar with a heavy right tail (a run of cloudy days), so it is bounded with a
conformal upper quantile rather than a point forecast.

ACI (Gibbs & Candes, NeurIPS 2021) adjusts the quantile level online from its
own misses, which gives a long-run miss rate close to alpha even though the
weather is not exchangeable. That miss rate is reported as a result, not
assumed: `coverage()` returns what actually happened.
"""
from typing import Dict, List, Optional

import numpy as np


class ReserveForecaster:
    def __init__(self, n_slots: int, horizon_slots: int, alpha: float, gamma: float,
                 window_days: int, eta_charge: float, eta_discharge: float,
                 warmup_days: int = 7, point_days: int = 3, conformal: bool = True):
        self.harvest = np.zeros(n_slots)
        self.essential = np.zeros(n_slots)
        self.horizon = int(horizon_slots)
        self.alpha = float(alpha)
        self.alpha_t = float(alpha)
        self.gamma = float(gamma)
        self.window = int(window_days)
        self.eta_c = eta_charge
        self.eta_d = eta_discharge
        self.warmup = warmup_days
        self.point_days = point_days
        self.conformal = conformal

        self._pending: List[int] = []            # sunsets not yet realised
        self._realised: List[float] = []
        self._scores: List[float] = []
        self._pred: Dict[int, tuple] = {}         # sunset slot -> (point, bound)
        self.misses: List[int] = []

    def record(self, t: int, harvest_j: float, essential_j: float):
        self.harvest[t] = harvest_j
        self.essential[t] = essential_j

    def mark_sunset(self, t: int):
        self._pending.append(int(t))

    def _required(self, s: int) -> float:
        seg = self.essential[s:s + self.horizon] / self.eta_d - self.eta_c * self.harvest[s:s + self.horizon]
        return float(max(0.0, np.cumsum(seg).max())) if len(seg) else 0.0

    def _realise(self, now: int):
        keep = []
        for s in self._pending:
            if s + self.horizon <= now:
                r = self._required(s)
                self._realised.append(r)
                if s in self._pred:
                    point, bound = self._pred.pop(s)
                    miss = int(r > bound)
                    self.misses.append(miss)
                    self._scores.append(r - point)
                    self.alpha_t += self.gamma * (self.alpha - miss)
            else:
                keep.append(s)
        self._pending = keep

    def point(self, now: int) -> Optional[float]:
        self._realise(now)
        if len(self._realised) < self.warmup:
            return None
        return float(max(self._realised[-self.point_days:]))

    def predict(self, sunset_slot: int, now: int) -> Optional[float]:
        """Upper bound on tonight's required reserve, or None during warm-up
        (the caller must then behave conservatively and keep the battery full)."""
        p = self.point(now)
        if p is None:
            return None
        if not self.conformal:
            self._pred[int(sunset_slot)] = (p, p)
            return p
        scores = np.asarray(self._scores[-self.window:])
        if len(scores) < 5:
            bound = 1.5 * p                   # few residuals yet: widen instead of trusting them
        elif self.alpha_t <= 0.0:
            bound = p + float(scores.max())   # ACI asks for more than the data can certify
        else:
            level = float(np.clip(1.0 - self.alpha_t, 0.0, 1.0))
            bound = p + float(np.quantile(scores, level, method="higher"))
        bound = max(bound, 0.0)
        self._pred[int(sunset_slot)] = (p, bound)
        return bound

    def coverage(self) -> Optional[float]:
        return None if not self.misses else 1.0 - float(np.mean(self.misses))
