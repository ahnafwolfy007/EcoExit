"""SunSched v4 candidates: v3 with the four faults found by reviewing its
held-out failures (Caltech cct_new and Serengeti S11, both now development data).

Each switch answers one fault and can be ablated on its own:

  two_tier  Fault: frames whose triage confidence clears the threshold are
            never refined, so a miscalibrated triage head under domain shift
            costs value even when energy is free (S11: -0.018 at 30-100 d).
            Fix: interruptible load, as in demand response. Confident frames
            are queued too, and refined only from energy the battery is about
            to refuse; otherwise they keep their triage label.
  reserve   Fault: at harvest ratio 1 v3 hoards a whole night's reserve while
            deferred frames expire unrefined (cct_new: a third of frames).
            "conformal" is v3. "survive" reserves only the sleep load up to
            sunrise (keep-alive priority, as in satellite safe mode). "merit"
            chooses between them by merit order, as a grid operator sheds
            load: value per joule of refining against value per joule of a
            capture, from running means of the node's calibrated estimates.
  al_min    Fault: the hold engages at ratio 1 after three good days and the
            wasted harvest is missed months later (S11 100 d: -0.22).
            Fix: array-to-load sizing (IEEE 1562): the hold may engage, and
            stays on, only while harvest since deployment exceeds al_min times
            the demand since deployment.
"""
import numpy as np

from sunsched.policies.v3 import SunSchedV3


class SunSchedV4(SunSchedV3):
    name = "sunsched4"

    def __init__(self, control_cfg, baseline_cfg, two_tier: bool = True, reserve: str = "survive",
                 al_min: float = 1.3, name: str = None, **kw):
        super().__init__(control_cfg, baseline_cfg, **kw)
        if reserve not in ("conformal", "survive", "merit"):
            raise ValueError(f"reserve {reserve!r}")
        self.name = name or "sunsched4"
        self.two_tier, self.reserve, self.al_min = two_tier, reserve, float(al_min)

    def reset(self, ctx):
        super().reset(ctx)
        self._al_h = self._al_d = 0.0     # harvest and demand since deployment, battery side (J)
        self._al_t = 0
        self._gain_sum = self._conf_sum = 0.0
        self._gain_n = self._conf_n = 0
        self._sleep_only = False          # this slot's reserve guards sleep only

    # -- array-to-load guard ------------------------------------------------------
    def array_to_load(self, t: int) -> float:
        fc, b = self.ctx.forecaster, self.ctx.cfg.battery
        a = self._al_t
        if t > a:
            self._al_h += b.eta_charge * float(fc.harvest[a:t].sum())
            self._al_d += float(fc.essential[a:t].sum() + self._refdem[a:t].sum()) / b.eta_discharge
            self._al_t = t
        return self._al_h / self._al_d if self._al_d > 0 else 0.0

    def _reservoir_hold(self, obs, judge=False, curtailed=False):
        super()._reservoir_hold(obs, judge, curtailed)
        if self.al_min > 0 and self.array_to_load(obs.t) < self.al_min and self._hold_on:
            self._hold_on, self._ceiling = False, 1.0

    # -- what to defer ------------------------------------------------------------
    def post_triage(self, obs, frame, pred, conf, gain):
        self._conf_sum += conf
        self._conf_n += 1
        if conf < self.theta:
            self._gain_sum += gain
            self._gain_n += 1
        elif self.two_tier:
            if self._cheap_wake() and self._surplus_now(obs):
                e = self.ctx.energy
                need = e.refine_j[self.refine_op] + (0.0 if obs.b_will_wake else e.wake_j)
                if self.usable_j(obs) >= need:
                    return "now"
            return "defer"
        return super().post_triage(obs, frame, pred, conf, gain)

    def _surplus_now(self, obs) -> bool:
        """Energy is about to be refused: the battery is at the level in force."""
        return obs.curtail_hint_j > 0 or obs.soc >= min(self._ceiling, 1.0) - 0.02

    # -- reserve ------------------------------------------------------------------
    def _refine_first(self, obs) -> bool:
        """Merit order: is a joule worth more in refinement than in a capture?"""
        if self._gain_n == 0 or self._conf_n == 0:
            return False
        e = self.ctx.energy
        per = e.refine_j[self.refine_op]
        batch = max(1.0, np.ceil(e.wake_j / per)) if self.coalesce else 1.0
        v_refine = (self._gain_sum / self._gain_n) / (per + e.wake_j / batch)
        v_capture = (self._conf_sum / self._conf_n) / (e.capture_j(True) + e.triage_j + e.store_j)
        return v_refine >= v_capture

    def target_j(self, obs):
        if self.reserve == "conformal" or self._bound is None:
            return super().target_j(obs)
        if self.reserve == "merit" and not self._refine_first(obs):
            return super().target_j(obs)
        night = 0.5 * self.ctx.slots_per_day
        sleep = 1.2 * night * self.ctx.energy.sleep_j / self.ctx.cfg.battery.eta_discharge
        # Never above what v3 would have kept.
        return min(super().target_j(obs), obs.floor_j + sleep)

    def usable_j(self, obs):
        if not obs.night or self.reserve == "conformal" or self._bound is None:
            return super().usable_j(obs)
        if self.reserve == "merit" and not self._refine_first(obs):
            return super().usable_j(obs)
        e, b = self.ctx.energy, self.ctx.cfg.battery
        here = obs.energy_j + b.eta_charge * obs.harvest_j - e.sleep_j / b.eta_discharge
        slots = max(self._sunrise_next(obs.t) - obs.t, 0)
        return max(0.0, here - obs.floor_j - 1.2 * slots * e.sleep_j / b.eta_discharge) * b.eta_discharge

    # -- batches: uncertain frames first, confident ones only from surplus ----------
    def refine_batch(self, obs, items):
        if not self.two_tier:
            return super().refine_batch(obs, items)
        urgent = [it for it in items if it.conf < self.theta]
        optional = [it for it in items if it.conf >= self.theta]
        chosen = []
        if urgent:      # waiting confident frames must not trigger the deadline rules
            age = obs.oldest_age_slots
            obs.oldest_age_slots = obs.t - min(it.t for it in urgent)
            chosen = super().refine_batch(obs, urgent)
            obs.oldest_age_slots = age
        if not optional or not self._surplus_now(obs):
            return chosen
        e = self.ctx.energy
        per = e.refine_j[self.refine_op]
        wake = 0.0 if (obs.b_will_wake or chosen) else e.wake_j
        usable = max(self.usable_j(obs), obs.curtail_hint_j * self.ctx.cfg.battery.eta_discharge)
        spent = len(chosen) * per + (0.0 if obs.b_will_wake or not chosen else e.wake_j)
        n = int(max(0.0, usable - spent - wake) // per)
        n = min(n, self.c.max_batch - len(chosen))
        if n <= 0:
            return chosen
        return chosen + sorted(optional, key=lambda it: it.t)[:n]
