"""SunSched v3: coalesce the expensive wake, spend the sun you cannot store,
and hold the charge only while the battery is full of idle energy.

Designed on the CCT20 development corpus only (five iterations, see
PREREGISTRATION_v2.md) after diagnosing why SunSched v1 lost: v1 refined a
frame only if a per-frame "expected gain" estimate cleared a threshold, and
that estimate is noise (the triage confidence barely varies, so the
calibration table says "no gain" for most frames). v1 therefore left 52-81%
of frames at the triage label. v3 keeps no per-frame value estimate. Every
uncertain frame is deferred and refined; the controller decides only WHEN to
wake the processor and HOW FULL to keep the battery.

Each mechanism is borrowed from a deployed real-world system:

1. Coalesced dispatch (Android Doze maintenance windows; TCP Nagle;
   economic order quantity). The processor wake is a fixed setup cost, so
   deferred frames are refined in batches. A batch is dispatched only when its
   refinement energy is at least the wake energy (n >= wake / refine), unless
   a frame is near its deadline, energy would otherwise be curtailed, it is
   the daily maintenance window (solar noon), or it is the last daylight
   chance before a frame expires. When a wake costs less than two
   refinements, frames are refined at once (TCP_NODELAY).

2. PV-overflow spending (solar diverters and home energy managers). In
   daylight the controller spends what is stored above tonight's reserve, or,
   if larger, the harvest forecast for the rest of the day (median of the last
   three days at the same clock times) that the battery could not store under
   the level in force. At night it spends only what the rest of the night
   does not need.

3. Reservoir hold (reservoir sizing and operation; charge-controller
   re-bulk). The hold keeps the charge at floor + max(sequent-peak storage
   requirement, tonight's conformal reserve) + margin. The sequent-peak
   requirement (Thomas & Burden 1963) is the largest cumulative deficit of
   demand (essential load plus the refinement the node asked for) over stored
   harvest seen so far. The hold engages after three consecutive surplus
   days; a day is surplus if the battery reached the level in force, or if it
   ended daylight at or above the hold level and held at least as much at
   sunrise as the day before (storage above the rule curve and still
   filling). It is released only when the sunrise energy falls below
   tonight's reserve, like a charge controller's re-bulk threshold.

4. Earliest-deadline-first order (real-time scheduling): oldest frames first.

No parameter is tuned per cell. The constants are the baselines' (lazy_theta,
lazy_soc_keep, ceiling_margin, min_ceiling, batch_interval_min, deadline_h),
plus a three-day history. Earlier development settings remain available as
named variants (policies sunsched3_iter1 ... sunsched3_iter4, sunsched3_rule).
"""
from collections import deque
from typing import List

import numpy as np

from sunsched.policies.base import Obs, Policy, QItem


class SunSchedV3(Policy):
    name = "sunsched3"

    def __init__(self, control_cfg, baseline_cfg, coalesce: bool = True, pv: bool = True,
                 hold: bool = True, name: str = None, history_days: int = 3,
                 forecast_scale: float = 1.0, hold_margin: float = None,
                 forecast_agg: str = "median", hold_level: str = "point",
                 hold_mode: str = "reservoir", spend_mode: str = "overflow",
                 hold_engage: str = "rise"):
        self.c = control_cfg
        self.theta = baseline_cfg.lazy_theta          # same deferral threshold as lazy_defer
        self.soc_keep = baseline_cfg.lazy_soc_keep    # lazy_defer's buffer, used when a night cannot be reserved
        self.margin = baseline_cfg.ceiling_margin if hold_margin is None else hold_margin
        self.coalesce, self.pv, self.hold = coalesce, pv, hold
        self.k = int(history_days)
        self.forecast_scale = float(forecast_scale)
        # Development iteration 2 (CCT20 only): the minimum over the last days
        # was too pessimistic at small storage, and holding at the conformal
        # bound kept the battery fuller than needed. Defaults are now the
        # median forecast and the point-forecast hold level of ceiling_now;
        # the iteration-1 settings remain available as named variants.
        self.forecast_agg = forecast_agg
        self.hold_level = hold_level
        # Development iteration 3 (CCT20 only): the hold judged surplus against
        # the conformal target, which sits above the hold level, so it kept
        # releasing itself; cheap wakes still lost to refining at once; and a
        # battery smaller than one night insisted on an unreachable reserve.
        self.iter3 = True
        # Development iteration 4 (CCT20 only). The hold released itself after
        # any dull day and then took weeks to re-engage at large storage, and at
        # small storage it capped a battery whose every joule was needed.
        #   hold_mode "reservoir": hold level from the sequent-peak storage
        #     requirement (reservoir sizing, Thomas & Burden 1963) over all
        #     history; released only when the sunrise energy falls below
        #     tonight's reserve (a charge controller's re-bulk threshold).
        #   spend_mode "overflow": in daylight, spend what is stored above
        #     tonight's target plus the forecast harvest that the battery could
        #     not store anyway (PV-surplus diversion); "forecast" is iteration
        #     3's rule, "stored" spends only what is stored above the target.
        self.hold_mode = hold_mode
        self.spend_mode = spend_mode
        # Development iteration 5 (CCT20 only). With the hold off, a day counted
        # as surplus only if the battery reached 98%, so a large battery that
        # started at 80% charged for months before the hold engaged.
        #   hold_engage "full": iteration 4 (the battery reached the level in force).
        #   hold_engage "rule": also a surplus day if the battery ended daylight at
        #     or above the level the hold would keep (reservoir rule curve:
        #     storage above the guide curve is idle).
        #   hold_engage "rise": "rule", and the battery held at least as much at
        #     this sunrise as at the previous one (the reservoir is filling).
        # Selected and frozen: "rise" (31 of 32 cells near-best on CCT20,
        # against 30 for "rule" and 29 for "full").
        if hold_engage not in ("full", "rule", "rise"):
            raise ValueError(f"hold_engage {hold_engage!r}")
        self.hold_engage = hold_engage
        if name:
            self.name = name

    # -- state -------------------------------------------------------------------
    def reset(self, ctx):
        super().reset(ctx)
        self._interval = max(int(round(self.c.batch_interval_min * 60 / ctx.slot_seconds)), 1)
        self._h = np.zeros(ctx.n_slots)              # observed harvest per slot
        self._bound = None                           # tonight's conformal reserve (J), None in warm-up
        self._surplus = deque(maxlen=self.k)         # per-day surplus flags
        self._hold_on = False
        self._day_curtailed = False
        self._ceiling = 1.0
        self._last_day = -1
        self._refdem = np.zeros(ctx.n_slots)         # refinement energy asked for, per slot
        self._sp_t = 0                               # next slot to fold into the sequent peak
        self._sp_k = 0.0                             # running cumulative deficit
        self._sp_max = 0.0                           # sequent-peak storage requirement so far
        self._dem_day = -1
        self._sunset_energy = None                   # stored energy at the last sunset (J)
        self._rise_energy = None                     # stored energy at the last sunrise (J)

    def _sunset(self, day: int, t: int) -> int:
        s = self.ctx.sunset[day] if 0 <= day < len(self.ctx.sunset) else -1
        return int(s) if s >= 0 else t + self.ctx.slots_per_day // 2

    def _sunrise_next(self, t: int) -> int:
        spd = self.ctx.slots_per_day
        d = t // spd
        for dd in (d, d + 1):
            if 0 <= dd < len(self.ctx.sunrise):
                s = int(self.ctx.sunrise[dd])
                if s > t:
                    return s
        return t + spd // 2

    def target_j(self, obs: Obs) -> float:
        """Energy the battery should hold at sunset: floor plus tonight's reserve.
        If the battery cannot hold a whole night's reserve, keep lazy_defer's
        buffer instead of insisting on an unreachable target."""
        if self._bound is None:
            return obs.floor_j + self.c.warmup_target_soc * obs.capacity_j
        if self.iter3 and obs.floor_j + self._bound >= obs.capacity_j:
            return max(obs.floor_j, self.soc_keep * obs.capacity_j)
        return min(obs.floor_j + self._bound, obs.capacity_j)

    def _cheap_wake(self) -> bool:
        """TCP_NODELAY: batching is not worth it when a wake costs less than two refinements."""
        e = self.ctx.energy
        return e.wake_j <= 2.0 * e.refine_j[self.refine_op]

    # -- daily bookkeeping and the learned hold -----------------------------------
    def on_sunrise(self, obs: Obs):
        # Close yesterday: surplus if it met its reserve at sunset and had energy to spare.
        closed, curtailed = self._last_day >= 0, self._day_curtailed
        judged_later = self.hold_mode == "reservoir" and self.hold_engage != "full"
        if closed and not judged_later:
            if self.iter3:      # surplus = the battery reached the level in force (phone-style)
                self._surplus.append(bool(curtailed))
            else:
                met = (self._sunset_energy or 0.0) >= self.target_j(obs) - 1e-9
                self._surplus.append(bool(met and curtailed))
        self._last_day = obs.day
        self._day_curtailed = False
        sunset = self._sunset(obs.day, obs.t)
        self._bound = self.ctx.forecaster.predict(int(sunset), obs.t)
        if self.hold_mode == "reservoir":
            self._reservoir_hold(obs, closed and judged_later, curtailed)
            self._rise_energy = obs.energy_j
            return
        self._hold_on = self.hold and len(self._surplus) == self.k and all(self._surplus)
        if self._hold_on and self._bound is not None:
            if self.hold_level == "point":
                point = self.ctx.forecaster.point(obs.t)
                level = obs.floor_j + (point if point is not None else self._bound)
            else:
                level = self.target_j(obs)
            self._ceiling = float(np.clip(level / obs.capacity_j + self.margin, self.c.min_ceiling, 1.0))
        else:
            self._ceiling = 1.0

    def _fold_sequent_peak(self, t: int):
        """Sequent-peak algorithm over every slot up to t-1: the running
        cumulative deficit of demand (essential load plus the refinement the
        node asked for) over stored harvest, and its maximum so far."""
        fc = self.ctx.forecaster
        cfg = self.ctx.cfg
        eta_c, eta_d = cfg.battery.eta_charge, cfg.battery.eta_discharge
        a, b = self._sp_t, max(t, self._sp_t)
        if b > a:
            net = (fc.essential[a:b] + self._refdem[a:b]) / eta_d - eta_c * fc.harvest[a:b]
            k, m = self._sp_k, self._sp_max
            for x in net:
                k = k + x if k + x > 0.0 else 0.0
                if k > m:
                    m = k
            self._sp_k, self._sp_max = k, m
            self._sp_t = b

    def _hold_ceiling(self, obs: Obs) -> float:
        """The state-of-charge ceiling the hold keeps: floor plus the larger of
        the sequent-peak requirement and tonight's reserve, plus the margin."""
        level = obs.floor_j + max(self._sp_max, self._bound)
        return float(np.clip(level / obs.capacity_j + self.margin, self.c.min_ceiling, 1.0))

    def _reservoir_hold(self, obs: Obs, judge: bool = False, curtailed: bool = False):
        self._fold_sequent_peak(obs.t)
        if judge:                                    # close yesterday against the rule curve
            flag = curtailed
            if not flag and self._bound is not None and self._sunset_energy is not None:
                rule_j = (self._hold_ceiling(obs) - 0.02) * obs.capacity_j
                flag = self._sunset_energy >= rule_j
                if flag and self.hold_engage == "rise":
                    flag = self._rise_energy is not None and obs.energy_j >= self._rise_energy
            self._surplus.append(bool(flag))
        if not self.hold or self._bound is None:
            self._hold_on, self._ceiling = False, 1.0
            return
        reserve = obs.floor_j + self._bound
        if self._hold_on and obs.energy_j < reserve:
            self._hold_on = False                    # re-bulk: tonight's reserve is threatened
            self._surplus.clear()
        elif not self._hold_on and len(self._surplus) == self.k and all(self._surplus):
            self._hold_on = True                     # surplus on each of the last k days
        self._ceiling = self._hold_ceiling(obs) if self._hold_on else 1.0

    def ceiling(self, obs: Obs) -> float:
        self._h[obs.t] = obs.harvest_j
        if obs.curtail_hint_j > 0 or obs.soc >= min(self._ceiling, 1.0) - 0.02:
            self._day_curtailed = True
        day = obs.day
        if obs.t == self._sunset(day, obs.t):
            self._sunset_energy = obs.energy_j
        return self._ceiling

    # -- what to defer ------------------------------------------------------------
    def post_triage(self, obs, frame, pred, conf, gain):
        if conf >= self.theta:
            return "final"
        e = self.ctx.energy
        self._refdem[obs.t] += e.refine_j[self.refine_op]
        if obs.day != self._dem_day:                 # one wake a day is part of the demand
            self._refdem[obs.t] += e.wake_j
            self._dem_day = obs.day
        if self.iter3 and self._cheap_wake():
            e = self.ctx.energy
            need = e.refine_j[self.refine_op] + (0.0 if obs.b_will_wake else e.wake_j)
            if self.usable_j(obs) >= need:
                return "now"
        return "defer"

    # -- how much may be spent now -----------------------------------------------
    def _remaining_harvest_j(self, t: int, until: int) -> float:
        """Pessimistic persistence forecast: the smallest of the last k days'
        harvest over the same clock interval, as stored (battery-side) energy."""
        spd = self.ctx.slots_per_day
        eta_c = self.ctx.cfg.battery.eta_charge
        sums = []
        for back in range(1, self.k + 1):
            a, b = t + 1 - back * spd, until - back * spd
            if a < 0:
                break
            sums.append(self._h[a:max(a, b)].sum())
        if not sums:
            return 0.0
        agg = min(sums) if self.forecast_agg == "min" else float(np.median(sums))
        return eta_c * self.forecast_scale * agg

    def usable_j(self, obs: Obs) -> float:
        """Load-side energy the controller may spend on refinement this slot."""
        e = self.ctx.energy
        cfg = self.ctx.cfg
        eta_c, eta_d = cfg.battery.eta_charge, cfg.battery.eta_discharge
        here = obs.energy_j + eta_c * obs.harvest_j - e.sleep_j / eta_d
        if not obs.night:
            sunset = self._sunset(obs.day, obs.t)
            slots = max(sunset - obs.t, 0)
            need_until_sunset = 1.2 * slots * e.sleep_j / eta_d
            stored = here - need_until_sunset - self.target_j(obs)
            if not self.pv or self.spend_mode == "stored":
                spend = stored if self.spend_mode == "stored" else here - self.target_j(obs)
            elif self.spend_mode == "overflow":
                # Spend what is stored above tonight's target, or, if more, the
                # forecast harvest that would overflow the level in force.
                rem = self._remaining_harvest_j(obs.t, sunset)
                cap = min(self._ceiling, 1.0) * obs.capacity_j
                overflow = min(here + rem - need_until_sunset - cap, here - obs.floor_j - need_until_sunset)
                spend = max(stored, overflow)
            else:
                rem = self._remaining_harvest_j(obs.t, sunset)
                after = here + rem - need_until_sunset - self.target_j(obs)
                # Cannot spend energy that is not in the battery yet.
                spend = min(after, here - obs.floor_j - need_until_sunset)
        else:
            rise = self._sunrise_next(obs.t)
            slots = max(rise - obs.t, 0)
            night_len = max(self.ctx.slots_per_day // 2, 1)
            reserve_left = (self._bound or 0.0) * min(slots / night_len, 1.0)
            spend = here - obs.floor_j - max(reserve_left, 1.2 * slots * e.sleep_j / eta_d)
        return max(0.0, spend) * eta_d

    # -- when and what to refine --------------------------------------------------
    def _maintenance_window(self, obs: Obs) -> bool:
        """One window per day around solar noon (Doze-style)."""
        if obs.night:
            return False
        d = obs.day
        if not (0 <= d < len(self.ctx.sunrise)) or self.ctx.sunrise[d] < 0 or self.ctx.sunset[d] < 0:
            return False
        noon = (int(self.ctx.sunrise[d]) + int(self.ctx.sunset[d])) // 2
        return abs(obs.t - noon) < max(self._interval // 2, 1)

    def _last_daylight_chance(self, obs: Obs) -> bool:
        """The oldest frame would expire before the next daylight window."""
        if obs.night or obs.oldest_age_slots <= 0:
            return False
        expires_at = obs.t - obs.oldest_age_slots + self.ctx.deadline_slots
        return expires_at < self._sunrise_next(self._sunset(obs.day, obs.t)) + 2 * self._interval

    def wants_batch(self, obs: Obs) -> bool:
        near_deadline = obs.oldest_age_slots >= self.ctx.deadline_slots - self._interval
        return (obs.t % self._interval == 0 or obs.curtail_hint_j > 0 or obs.b_will_wake
                or near_deadline or self._maintenance_window(obs))

    def refine_batch(self, obs: Obs, items: List[QItem]) -> List[QItem]:
        e = self.ctx.energy
        per = e.refine_j[self.refine_op]
        wake = 0.0 if obs.b_will_wake else e.wake_j
        usable = self.usable_j(obs)
        if obs.curtail_hint_j > 0:
            usable = max(usable, obs.curtail_hint_j * self.ctx.cfg.battery.eta_discharge)
        n_afford = int(max(0.0, usable - wake) // per)
        if n_afford <= 0:
            return []
        items = sorted(items, key=lambda it: it.t)          # earliest deadline first
        near = sum(1 for it in items if obs.t - it.t >= self.ctx.deadline_slots - self._interval)
        free = obs.curtail_hint_j > 0 or obs.b_will_wake
        window = self._maintenance_window(obs) or self._last_daylight_chance(obs)
        if self.coalesce and not free and near == 0 and not window:
            n_min = int(np.ceil(wake / per)) if per > 0 else 1   # wake <= batch refinement energy
            if min(len(items), n_afford) < n_min:
                return []                                    # wait: the batch is not worth a wake yet
        return items[:min(n_afford, self.c.max_batch, len(items))]
