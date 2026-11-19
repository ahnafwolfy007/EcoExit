"""Baselines, each standing for a published class of controller.

They are re-implementations of the *decision rule* of each class, run inside
the same simulator with the same classifier, so a difference in the results
table is a difference in decisions and nothing else.
"""
from typing import List

import numpy as np

from sunsched.policies.base import Obs, Policy, QItem


def reserve_ceiling(ctx, obs: Obs, bound: str, margin: float, min_ceiling: float) -> float:
    """Charge ceiling for today: tonight's reserve need, plus a fixed margin.

    bound="point" uses the point forecast (the published rule); "conformal"
    uses the conformal upper bound at the configured risk level instead.
    During forecaster warm-up the battery is allowed to fill.
    """
    fc = ctx.forecaster
    if bound == "conformal":
        day = obs.t // ctx.slots_per_day
        sunset = ctx.sunset[day] if 0 <= day < len(ctx.sunset) else -1
        if sunset < 0:
            sunset = obs.t + ctx.slots_per_day // 2
        need = fc.predict(int(sunset), obs.t)
    else:
        need = fc.point(obs.t)
    if need is None:
        return 1.0
    return float(np.clip((obs.floor_j + need) / obs.capacity_j + margin, min_ceiling, 1.0))


class AlwaysNow(Policy):
    """Run the full model on every capture, immediately; keep the battery full.
    The accuracy-first default of an unmanaged node."""
    name = "always_now"

    def pre_capture(self, obs, frame):
        return "direct"


class TriageOnly(Policy):
    """Never wake the application processor: the cheapest possible node."""
    name = "triage_only"


class EnergyAwareEarlyExit(Policy):
    """Per-frame, energy-aware early exit, decided at capture time.

    Stands for Bullo et al. (MLSP 2023; arXiv 2411.02471) and the older
    battery-bucket lookup tables (ePerceptive, SenSys 2020): after the cheap
    exit, continue to the deep exits now if confidence is below a threshold
    that falls as the battery drains. It never defers and keeps the battery full.
    """
    name = "ee_now"

    def __init__(self, theta_hi: float, soc_hi: float, soc_lo: float):
        self.theta_hi, self.soc_hi, self.soc_lo = theta_hi, soc_hi, soc_lo

    def threshold(self, soc: float) -> float:
        x = (soc - self.soc_lo) / max(self.soc_hi - self.soc_lo, 1e-9)
        return self.theta_hi * float(np.clip(x, 0.0, 1.0))

    def post_triage(self, obs, frame, pred, conf, gain):
        return "now" if conf < self.threshold(obs.soc) else "final"

    def now_op(self, obs):
        return "full" if obs.soc >= self.soc_hi else "lite"


class LazyDeferral(Policy):
    """Deadline-aware lazy scheduling (Moser et al., 2006/2007), adapted.

    Uncertain frames are deferred and refined first-in first-out whenever the
    battery is above a switch-on level in daylight, keeping a fixed buffer. It
    defers like the proposed policy but has no model of which frames are worth
    refining, of battery ageing, or of risk, and it keeps the battery full.
    """
    name = "lazy_defer"

    def __init__(self, theta: float, soc_on: float, soc_keep: float, interval_slots: int,
                 daylight_only: bool = True, name: str = None):
        self.theta, self.soc_on, self.soc_keep = theta, soc_on, soc_keep
        self.interval = max(int(interval_slots), 1)
        self.daylight_only = daylight_only
        if name:
            self.name = name

    def post_triage(self, obs, frame, pred, conf, gain):
        return "defer" if conf < self.theta else "final"

    def wants_batch(self, obs):
        if self.daylight_only and obs.night:
            return False
        return obs.soc >= self.soc_on and (
            obs.t % self.interval == 0 or obs.curtail_hint_j > 0 or obs.b_will_wake)

    def refine_batch(self, obs, items):
        e = self.ctx.energy
        eta_d = self.ctx.cfg.battery.eta_discharge
        usable = (obs.energy_j - self.soc_keep * obs.capacity_j) * eta_d
        if not obs.b_will_wake:
            usable -= e.wake_j
        n = int(max(0.0, usable) // e.refine_j[self.refine_op])
        items = sorted(items, key=lambda it: it.t)
        return items[:n]


class ChargeCeilingEarlyExit(EnergyAwareEarlyExit):
    """Energy-aware early exit plus a charge ceiling set from a point forecast.

    Stands for predicted-harvest charge limiting for IoT battery lifespan
    (IEICE Trans. E103-D, 2020): cap the charge at the forecast need plus a
    fixed margin. No deferral, no calibrated risk.
    """
    name = "ceiling_now"

    def __init__(self, theta_hi, soc_hi, soc_lo, margin: float, min_ceiling: float,
                 bound: str = "point", name: str = None):
        super().__init__(theta_hi, soc_hi, soc_lo)
        self.margin, self.min_ceiling = margin, min_ceiling
        self.bound = bound
        if name:
            self.name = name
        self._ceiling = 1.0

    def reset(self, ctx):
        super().reset(ctx)
        self._ceiling = 1.0

    def on_sunrise(self, obs):
        self._ceiling = reserve_ceiling(self.ctx, obs, self.bound, self.margin, self.min_ceiling)

    def ceiling(self, obs):
        return self._ceiling

    def threshold(self, soc: float) -> float:
        # Judge the battery relative to its own ceiling. Against absolute
        # thresholds a node capped at, say, 40% would never look "full" and
        # would stop refining altogether -- a strawman, not the published idea.
        return super().threshold(soc / max(self._ceiling, 1e-9))

    def now_op(self, obs):
        return "full" if obs.soc / max(self._ceiling, 1e-9) >= self.soc_hi else "lite"


class DeferCeiling(LazyDeferral):
    """Lazy deferral to daylight combined with a charge ceiling (paper revision).

    The direct "why not both" controller: uncertain frames are deferred and
    refined in daylight batches exactly as in LazyDeferral, and the battery is
    capped as in ChargeCeilingEarlyExit. Every parameter is the default of the
    two baselines it combines; nothing was tuned for it. Battery levels are
    judged against the ceiling, as in ChargeCeilingEarlyExit, so a node capped
    at 40% still refines.
    """
    name = "defer_ceiling"

    def __init__(self, theta, soc_on, soc_keep, interval_slots, margin, min_ceiling,
                 bound: str = "point", name: str = None):
        super().__init__(theta, soc_on, soc_keep, interval_slots, daylight_only=True, name=name)
        self.margin, self.min_ceiling, self.bound = margin, min_ceiling, bound
        self._ceiling = 1.0

    def reset(self, ctx):
        super().reset(ctx)
        self._ceiling = 1.0

    def on_sunrise(self, obs):
        self._ceiling = reserve_ceiling(self.ctx, obs, self.bound, self.margin, self.min_ceiling)

    def ceiling(self, obs):
        return self._ceiling

    def wants_batch(self, obs):
        if obs.night:
            return False
        rel = obs.soc / max(self._ceiling, 1e-9)
        return rel >= self.soc_on and (
            obs.t % self.interval == 0 or obs.curtail_hint_j > 0 or obs.b_will_wake)

    def refine_batch(self, obs, items):
        e = self.ctx.energy
        eta_d = self.ctx.cfg.battery.eta_discharge
        usable = (obs.energy_j - self.soc_keep * self._ceiling * obs.capacity_j) * eta_d
        if not obs.b_will_wake:
            usable -= e.wake_j
        n = int(max(0.0, usable) // e.refine_j[self.refine_op])
        return sorted(items, key=lambda it: it.t)[:n]
