"""SunSched: triage now, refine when the sun pays for it, and do not bank the sun.

Three mechanisms, each ablatable:

1. Deferral. Every capture is triaged by the cheap exit when it happens. A
   frame whose expected refinement gain is worth having is stored, not
   refined on the spot. On CCT20 most events arrive at night, when refining
   means waking the application processor on battery power.

2. Batched refinement from surplus only. Deferred frames are refined in
   batches, one processor wake per batch, using only energy above tonight's
   reserve. Frames are taken in order of value gain times deadline urgency,
   and a batch runs only if its total gain beats its energy at the current
   price. Below the reserve the price rises steeply, so the night's capture
   capability is never traded for accuracy.

3. A risk-controlled charge ceiling. The reserve is a conformal upper bound on
   the energy the node needs to get through the coming night (see
   env/forecast.py), with miss rate about alpha. The battery is charged to
   that reserve plus a margin, not to 100%, because a lithium cell held full
   in a hot enclosure ages fastest. Charge above the ceiling is either spent
   on refinement or curtailed.

The price is a drift-plus-penalty form: it grows with the deficit of stored
energy below the reserve target. Its theoretical properties are not claimed
here; the code implements the rule, and the experiments measure it.
"""
from typing import List

import numpy as np

from sunsched.policies.base import Obs, Policy, QItem


class SunSched(Policy):
    name = "sunsched"

    def __init__(self, control_cfg, defer: bool = True, ceiling_on: bool = True,
                 conformal: bool = True, refine_op: str = "full", name: str = None):
        self.c = control_cfg
        self.defer = defer
        self.ceiling_on = ceiling_on
        self.uses_conformal = conformal         # read by the runner when it builds the forecaster
        self.refine_op = refine_op
        if name:
            self.name = name

    def reset(self, ctx):
        super().reset(ctx)
        self._ceiling = 1.0
        self._target_j = None
        self._interval = max(int(round(self.c.batch_interval_min * 60 / ctx.slot_seconds)), 1)

    # -- reserve and ceiling -----------------------------------------------------
    def on_sunrise(self, obs: Obs):
        day = obs.day
        sunset = self.ctx.sunset[day] if 0 <= day < len(self.ctx.sunset) else -1
        if sunset < 0:
            sunset = obs.t + self.ctx.slots_per_day // 2
        reserve = self.ctx.forecaster.predict(int(sunset), obs.t)
        if reserve is None:
            self._target_j = None
            self._ceiling = 1.0
            return
        self._target_j = obs.floor_j + reserve
        if self.ceiling_on:
            self._ceiling = float(np.clip(self._target_j / obs.capacity_j + self.c.reserve_margin,
                                          self.c.min_ceiling, 1.0))
        else:
            self._ceiling = 1.0

    def target_j(self, obs: Obs) -> float:
        if self._target_j is None:
            return self.c.warmup_target_soc * obs.capacity_j
        return self._target_j

    def ceiling(self, obs: Obs) -> float:
        return self._ceiling

    def price(self, obs: Obs) -> float:
        """Value per joule. Low above the reserve, steep below it."""
        deficit = max(0.0, self.target_j(obs) - obs.energy_j) / obs.capacity_j
        return self.c.base_price + self.c.scarcity_beta * deficit

    # -- per-frame decision ------------------------------------------------------
    def post_triage(self, obs, frame, pred, conf, gain):
        if gain < self.c.min_gain:
            return "final"
        if self.defer:
            return "defer"
        e = self.ctx.energy
        cost = e.refine_j[self.refine_op] + (0.0 if obs.b_will_wake else e.wake_j)
        return "now" if self.c.V * gain > self.price(obs) * cost else "final"

    # -- batches -----------------------------------------------------------------
    def wants_batch(self, obs: Obs) -> bool:
        if not self.defer:
            return False
        near_deadline = obs.oldest_age_slots >= self.ctx.deadline_slots - self._interval
        return (obs.t % self._interval == 0 or obs.curtail_hint_j > 0.0
                or obs.b_will_wake or near_deadline)

    def refine_batch(self, obs: Obs, items: List[QItem]) -> List[QItem]:
        cfg = self.ctx.cfg
        e = self.ctx.energy
        eta_c, eta_d = cfg.battery.eta_charge, cfg.battery.eta_discharge
        sleep_b = e.sleep_j / eta_d
        # Load-side energy available above tonight's reserve, counting this
        # slot's harvest. Anything the ceiling would curtail is inside this.
        usable = (obs.energy_j + eta_c * obs.harvest_j - sleep_b - self.target_j(obs)) * eta_d
        wake = 0.0 if obs.b_will_wake else e.wake_j
        per_frame = e.refine_j[self.refine_op]
        if usable < wake + per_frame:
            return []

        lam = self.price(obs)
        deadline = max(self.ctx.deadline_slots, 1)
        age = np.array([obs.t - it.t for it in items], dtype=np.float64)
        gain = np.array([it.gain for it in items], dtype=np.float64)
        priority = self.c.V * gain * (1.0 + self.c.urgency * np.clip(age / deadline, 0, 1) ** 2)
        order = np.argsort(-priority, kind="stable")

        n_afford = int((usable - wake) // per_frame)
        chosen, total = [], 0.0
        for i in order[:min(n_afford, self.c.max_batch)]:
            if priority[i] <= lam * per_frame:
                break
            chosen.append(items[i])
            total += priority[i]
        if not chosen or total <= lam * (wake + per_frame * len(chosen)):
            return []
        return chosen
