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
                 warmup_days: int = 7, point_days: int = 3, conformal: bool = True,
                 shadow_alphas=()):
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

        # Shadow reserve methods (paper revision). At every sunset where the
        # policy asks for a bound, the same information is also turned into a
        # bound by four alternative rules, at several risk levels. They are
        # scored on the same realised nights but never reach the policy, so the
        # simulated deployment is unchanged. Methods:
        #   point      the point forecast alone (no margin)
        #   split      split-conformal: point + fixed (1-a) quantile of past residuals, no adaptation
        #   aci        Adaptive Conformal Inference at level a (the method used, run at other a)
        #   empirical  (1-a) quantile of past realised requirements, no point forecast
        self.shadow_alphas = tuple(float(a) for a in shadow_alphas)
        self._sh_alpha_t = {a: a for a in self.shadow_alphas}
        self._sh_pred: Dict[int, Dict[str, float]] = {}
        self._sh_miss: Dict[str, List[int]] = {}
        self._sh_bound: Dict[str, List[float]] = {}
        self._sh_req: List[float] = []

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
                if s in self._sh_pred:
                    self._shadow_score(self._sh_pred.pop(s), r)
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
        if self.shadow_alphas:
            self._shadow_predict(int(sunset_slot), p)
        return bound

    # -- shadow methods ---------------------------------------------------------
    def _aci_bound(self, p: float, alpha_t: float) -> float:
        scores = np.asarray(self._scores[-self.window:])
        if len(scores) < 5:
            return 1.5 * p
        if alpha_t <= 0.0:
            return p + float(scores.max())
        return p + float(np.quantile(scores, float(np.clip(1.0 - alpha_t, 0.0, 1.0)), method="higher"))

    def _shadow_predict(self, sunset_slot: int, p: float):
        scores = np.asarray(self._scores[-self.window:])
        past = np.asarray(self._realised[-self.window:])
        b = {"point": max(p, 0.0)}
        for a in self.shadow_alphas:
            b[f"aci_{a:g}"] = max(self._aci_bound(p, self._sh_alpha_t[a]), 0.0)
            if len(scores) < 5:
                b[f"split_{a:g}"] = 1.5 * p
            else:
                b[f"split_{a:g}"] = max(p + float(np.quantile(scores, 1.0 - a, method="higher")), 0.0)
            b[f"empirical_{a:g}"] = float(np.quantile(past, 1.0 - a, method="higher")) if len(past) else p
        self._sh_pred[sunset_slot] = b

    def _shadow_score(self, bounds: Dict[str, float], r: float):
        self._sh_req.append(r)
        for k, bound in bounds.items():
            miss = int(r > bound)
            self._sh_miss.setdefault(k, []).append(miss)
            self._sh_bound.setdefault(k, []).append(bound)
            if k.startswith("aci_"):
                a = float(k[4:])
                self._sh_alpha_t[a] += self.gamma * (a - miss)

    def shadow_summary(self) -> Dict[str, float]:
        """cov_<method>: share of scored nights the bound covered.
        size_<method>: mean bound over mean realised requirement (how much
        reserve the method asks for; lower is cheaper for the battery)."""
        if not self._sh_req:
            return {}
        req = float(np.mean(self._sh_req))
        out = {"shadow_nights": float(len(self._sh_req)), "shadow_req_mean_j": req}
        for k in self._sh_miss:
            out[f"cov_{k}"] = 1.0 - float(np.mean(self._sh_miss[k]))
            out[f"size_{k}"] = float(np.mean(self._sh_bound[k])) / max(req, 1e-12)
        return out

    def coverage(self) -> Optional[float]:
        return None if not self.misses else 1.0 - float(np.mean(self.misses))
