"""SunSched v5 candidates (PLAN_V2 section 3). Development only.

v5 is the full v4 (two-tier queue, survival reserve, array-to-load guard) with
four switches and two knobs:

  backoff  Fault it answers: on Serengeti S10 the hold lowers the ceiling, a
           burst of activity empties the battery, and captures are lost (11 %
           against 2 % without the hold). Rule: after a shortage during or
           soon after a hold, release it and do not hold again for B days; B
           doubles at every shortage (binary exponential backoff, as in
           Ethernet and TCP retransmission).
  late     Paid wakes as late as is safe (Proposition 1): no daily noon window
           and no break-even batch. A wake is bought at the last daylight
           chance before the oldest urgent frame expires, or earlier when the
           queue already fills what one wake may spend.
  buffer   Fault it answers: at 0.5 to 2 days of storage, and under any low cap,
           the night reserve is as large as what the battery may hold, so
           nothing is left to spend and deferred frames expire (S10: +0.055
           against lazy deferral's +0.103 at a cap of 0.5). Rule: the energy
           kept at sunset is never more than the fixed buffer of lazy deferral,
           taken as the same share of the range between floor and ceiling.
  fallback Fault it answers: with the buffer rule S10 still loses 0.04 value to
           lazy deferral at low caps and small storage, because daylight
           spending also reserves the sleep energy until sunset, which the sun
           pays anyway. Rule: while the night's sleep energy does not fit into
           the buffer, uncertain frames are refined by the rule of lazy deferral
           (daylight, charge above its switch-on level, its buffer kept), with
           levels relative to the ceiling in force.
  margin   Charge kept above the hold level.
  cap      The knob of the value-life front: a charge ceiling that always applies
           (the hold may go below it).
"""
from typing import List

import numpy as np

from sunsched.policies.base import Obs, QItem
from sunsched.policies.baselines import LazyDeferral
from sunsched.policies.v3 import SunSchedV3
from sunsched.policies.v4 import SunSchedV4


class _LateWakes(SunSchedV3):
    """SunSchedV3.refine_batch with the paid-wake rule exchanged (see `late`)."""
    late = True

    def refine_batch(self, obs: Obs, items: List[QItem]) -> List[QItem]:
        if self._small(obs):
            return self._lazy_batch(obs, items)
        if not self.late:
            return super().refine_batch(obs, items)
        e = self.ctx.energy
        per = e.refine_j[self.refine_op]
        wake = 0.0 if obs.b_will_wake else e.wake_j
        usable = self.usable_j(obs)
        if obs.curtail_hint_j > 0:
            usable = max(usable, obs.curtail_hint_j * self.ctx.cfg.battery.eta_discharge)
        n_afford = int(max(0.0, usable - wake) // per)
        if n_afford <= 0:
            return []
        items = sorted(items, key=lambda it: it.t)
        near = obs.t - items[0].t >= self.ctx.deadline_slots - self._interval
        free = obs.curtail_hint_j > 0 or obs.b_will_wake or self._refused_today(obs) >= wake + per
        full = not obs.night and len(items) >= n_afford
        if not (near or free or full or self._last_daylight_chance(obs)):
            return []
        return items[:min(n_afford, self.c.max_batch)]

    def _small(self, obs: Obs) -> bool:
        return False

    def _lazy_batch(self, obs: Obs, items: List[QItem]) -> List[QItem]:
        """LazyDeferral's rule, with its two levels taken relative to the ceiling in force."""
        level = min(self._ceiling, 1.0)
        if obs.night or obs.soc < self.soc_on * level:
            return []
        if not (obs.t % self._interval == 0 or obs.curtail_hint_j > 0 or obs.b_will_wake):
            return []
        e = self.ctx.energy
        usable = (obs.energy_j - self.soc_keep * level * obs.capacity_j) * self.ctx.cfg.battery.eta_discharge
        if not obs.b_will_wake:
            usable -= e.wake_j
        n = int(max(0.0, usable) // e.refine_j[self.refine_op])
        return sorted(items, key=lambda it: it.t)[:min(n, self.c.max_batch)]

    def _refused_today(self, obs: Obs) -> float:
        """Load-side energy that the forecast says today's sun will offer beyond the level in force."""
        if obs.night:
            return 0.0
        e, b = self.ctx.energy, self.ctx.cfg.battery
        sunset = self._sunset(obs.day, obs.t)
        need = 1.2 * max(sunset - obs.t, 0) * e.sleep_j / b.eta_discharge
        over = obs.energy_j + self._remaining_harvest_j(obs.t, sunset) - need - min(self._ceiling, 1.0) * obs.capacity_j
        return max(0.0, over) * b.eta_discharge


class SunSchedV5(SunSchedV4, _LateWakes):
    name = "sunsched5"

    def __init__(self, control_cfg, baseline_cfg, backoff: bool = True, late: bool = True, margin: float = 0.10,
                 cap: float = 1.0, buffer: bool = False, fallback: bool = False, backoff_days: int = 14, two_tier: bool = True, reserve: str = "survive", al_min: float = 1.3,
                 name: str = None, **kw):
        super().__init__(control_cfg, baseline_cfg, two_tier=two_tier, reserve=reserve, al_min=al_min,
                         name=name or "sunsched5", hold_margin=margin, **kw)
        self.backoff, self.late, self.backoff0, self.cap = backoff, late, int(backoff_days), float(cap)
        self.buffer, self.fallback = bool(buffer), bool(fallback)
        self.soc_on = baseline_cfg.lazy_soc_on

    def reset(self, ctx):
        super().reset(ctx)
        self._short = False
        self._hold_until = -1          # no hold before this day
        self._held_last = -10 ** 6     # last day a hold was on
        self._b = self.backoff0
        self._ceiling = self.cap

    def _buffer_j(self, obs: Obs) -> float:
        """Lazy deferral's buffer as a share of the range between floor and ceiling (J above the floor)."""
        floor = obs.floor_j / obs.capacity_j
        share = (self.soc_keep - floor) / (1.0 - floor)
        return share * max(min(self._ceiling, 1.0) - floor, 0.0) * obs.capacity_j

    def _night_sleep_j(self) -> float:
        return 1.2 * 0.5 * self.ctx.slots_per_day * self.ctx.energy.sleep_j / self.ctx.cfg.battery.eta_discharge

    def _small(self, obs: Obs) -> bool:
        return self.fallback and self._bound is not None and self._night_sleep_j() > self._buffer_j(obs)

    def target_j(self, obs: Obs) -> float:
        t = super().target_j(obs)
        if not self.buffer or self._bound is None:
            return t
        return min(t, obs.floor_j + self._buffer_j(obs))

    def _note_shortage(self, obs: Obs):
        """This slot cannot pay for a night capture and its triage."""
        e, b = self.ctx.energy, self.ctx.cfg.battery
        here = obs.energy_j + b.eta_charge * obs.harvest_j - e.sleep_j / b.eta_discharge - obs.floor_j
        if here < (e.capture_j(True) + e.triage_j + e.store_j) / b.eta_discharge:
            self._short = True

    def ceiling(self, obs: Obs) -> float:
        if self.backoff:
            self._note_shortage(obs)
        return super().ceiling(obs)

    def _reservoir_hold(self, obs, judge=False, curtailed=False):
        super()._reservoir_hold(obs, judge, curtailed)
        self._ceiling = min(self._ceiling, self.cap)
        if not self.backoff:
            return
        if self._short and obs.day - self._held_last <= 30:
            self._hold_until = obs.day + self._b
            self._b = min(2 * self._b, 365)
            self._held_last = -10 ** 6
        self._short = False
        if obs.day < self._hold_until and self._hold_on:
            self._hold_on, self._ceiling = False, self.cap
        if self._hold_on:
            self._held_last = obs.day


class LazyCeiling(LazyDeferral):
    """Lazy deferral under a fixed charge ceiling; levels are judged against the
    ceiling so that a low one does not stop refinement. The static competitor
    for the value-life front."""

    def __init__(self, theta, soc_on, soc_keep, interval_slots, level: float, name: str):
        super().__init__(theta, soc_on * level, soc_keep * level, interval_slots, name=name)
        self.level = float(level)

    def ceiling(self, obs: Obs) -> float:
        return self.level
