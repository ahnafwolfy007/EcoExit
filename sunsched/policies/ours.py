"""SunSched v2: refine now if the surplus pays for it, otherwise wait for the sun.

Version 1 always deferred, and chose what to refine from the species head's
top-1 confidence. On CCT20 it failed its kill test, for two reasons found
afterwards. That confidence was almost constant, so its refinement choices
were noise. And with a large battery there is nothing to gain from waiting,
so deferral only added latency.

Version 2 keeps one rule and lets the battery decide which behaviour applies:

1. Triage every capture on the low-power path. Its expected refinement gain
   comes from the triage *detector's* animal probability; frames very likely
   empty are finalised at triage.

2. The reserve gate. Tonight's reserve is a conformal upper bound on the
   energy needed to keep capturing until harvest resumes (env/forecast.py).
   A frame worth refining is refined immediately if the energy above that
   reserve covers it (sharing one processor wake with anything else refined
   in the same slot). Otherwise it is deferred.
   - Large battery: there is almost always surplus, so it refines now and
     behaves like refine-immediately with a charge ceiling.
   - Small battery: at night there is no surplus, so it defers and the night's
     energy goes to capturing, not classifying.

3. Deferred frames are refined in batches from daytime surplus, one wake per
   batch, highest gain times deadline urgency first.

4. The charge ceiling is the reserve plus a margin, so a large battery is not
   held full in a hot enclosure. With a small battery the reserve is most of
   the capacity and the ceiling barely binds.

Every mechanism can be switched off for the ablations (experiment.make_policy).
"""
from typing import List

import numpy as np

from sunsched.policies.base import Obs, Policy, QItem


class SunSched(Policy):
    name = "sunsched"

    def __init__(self, control_cfg, gate: bool = True, defer: bool = True,
                 ceiling_on: bool = True, conformal: bool = True,
                 refine_op: str = "full", name: str = None):
        self.c = control_cfg
        self.gate = gate                  # False: always defer (v1 behaviour)
        self.defer = defer                # False: refine now if affordable, else keep triage
        self.ceiling_on = ceiling_on
        self.uses_conformal = conformal   # read by the runner when it builds the forecaster
        self.refine_op = refine_op
        if name:
            self.name = name

    def reset(self, ctx):
        super().reset(ctx)
        self._ceiling = 1.0
        self._target_j = None
        self._interval = max(int(round(self.c.batch_interval_min * 60 / ctx.slot_seconds)), 1)
        self._slot = -1
        self._committed_j = 0.0           # energy promised to refine-now frames this slot
        self._now_in_slot = 0
        self._rem_slot, self._rem_value = -1, 0.0

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
        self._target_j = min(obs.floor_j + reserve, obs.capacity_j)
        self._ceiling = (float(np.clip(self._target_j / obs.capacity_j + self.c.reserve_margin,
                                       self.c.min_ceiling, 1.0)) if self.ceiling_on else 1.0)

    def target_j(self, obs: Obs) -> float:
        if self._target_j is None:
            return self.c.warmup_target_soc * obs.capacity_j
        return self._target_j

    def ceiling(self, obs: Obs) -> float:
        return self._ceiling

    def price(self, obs: Obs) -> float:
        """Value per joule: low above the reserve, steep below it."""
        deficit = max(0.0, self.target_j(obs) - obs.energy_j) / obs.capacity_j
        return self.c.base_price + self.c.scarcity_beta * deficit

    def remaining_harvest_lb_j(self, obs: Obs) -> float:
        """Causal lower bound on the harvest still to come before today's sunset.

        The reserve has to be in the battery at sunset, not all day. Before
        sunset, energy that today's remaining sun will replace is also surplus.
        Without this term, a battery smaller than one night's reserve never has
        any surplus at all, so deferred frames are never refined and simply
        expire -- which is what the synthetic smoke test showed. The bound is the
        smallest harvest over the same clock hours on the previous 3 days, so it
        uses only the past.
        """
        if obs.night:
            return 0.0
        if obs.t == getattr(self, "_rem_slot", -1):
            return self._rem_value
        spd = self.ctx.slots_per_day
        day = obs.day
        sunset = self.ctx.sunset[day] if 0 <= day < len(self.ctx.sunset) else -1
        value = 0.0
        if sunset >= 0 and obs.t < sunset:
            start, end = obs.t % spd, sunset % spd
            past = [self.ctx.forecaster.harvest[(day - k) * spd + start:(day - k) * spd + end + 1].sum()
                    for k in (1, 2, 3) if day - k >= 0]
            value = float(min(past)) if past else 0.0
        self._rem_slot, self._rem_value = obs.t, value
        return value

    def surplus_j(self, obs: Obs) -> float:
        """Load-side energy available now above what must be in the battery at
        sunset: stored energy plus this slot's harvest, plus (in daylight) the
        lower bound on the rest of today's harvest, minus tonight's reserve."""
        cfg = self.ctx.cfg
        eta_c = cfg.battery.eta_charge
        sleep_b = self.ctx.energy.sleep_j / cfg.battery.eta_discharge
        above = (obs.energy_j + eta_c * obs.harvest_j - sleep_b - self.target_j(obs)
                 + eta_c * self.remaining_harvest_lb_j(obs))
        # Cannot spend more than is physically stored plus this slot's harvest.
        available = obs.energy_j + eta_c * obs.harvest_j - sleep_b - obs.floor_j
        return max(0.0, min(above, available)) * cfg.battery.eta_discharge

    # -- per-frame decision ------------------------------------------------------
    def post_triage(self, obs, frame, pred, conf, gain, p_animal):
        if gain < self.c.min_gain:
            return "final"
        if obs.t != self._slot:
            self._slot, self._committed_j, self._now_in_slot = obs.t, 0.0, 0
        if self.gate or not self.defer:
            e = self.ctx.energy
            cost = e.refine_j[self.refine_op] + (0.0 if self._now_in_slot or obs.b_will_wake else e.wake_j)
            worth = self.c.V * gain > self.price(obs) * cost
            if worth and self.surplus_j(obs) - self._committed_j >= cost:
                self._committed_j += cost
                self._now_in_slot += 1
                return "now"
        return "defer" if self.defer else "final"

    # -- batches -----------------------------------------------------------------
    def wants_batch(self, obs: Obs) -> bool:
        if not self.defer:
            return False
        near_deadline = obs.oldest_age_slots >= self.ctx.deadline_slots - self._interval
        return (obs.t % self._interval == 0 or obs.curtail_hint_j > 0.0
                or obs.b_will_wake or near_deadline)

    def refine_batch(self, obs: Obs, items: List[QItem]) -> List[QItem]:
        e = self.ctx.energy
        usable = self.surplus_j(obs) - (self._committed_j if obs.t == self._slot else 0.0)
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
