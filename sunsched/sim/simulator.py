"""Slot-by-slot simulation of one camera over one weather window.

Every policy runs through this same loop, over the same real captures and the
same real weather, so the only thing that differs between rows of a results
table is the decision rule.

Energy accounting is on the battery side: a load of x joules costs x/eta_d of
stored energy and harvest h adds eta_c*h. An action is only taken if it is
affordable without going below the floor; the always-on sleep draw is the one
exception, and if even that cannot be paid the node is dead for the slot and
misses whatever arrives.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from sunsched.policies.base import Obs, Policy, QItem, RunContext

DROPPED, TRIAGE, REFINED_NOW, REFINED_LATER, UNCLASSIFIED = 0, 1, 2, 3, 4
ROUTE_NAMES = {DROPPED: "dropped", TRIAGE: "triage", REFINED_NOW: "refined_now",
               REFINED_LATER: "refined_later", UNCLASSIFIED: "unclassified"}


@dataclass
class SimResult:
    frames: np.ndarray            # evaluation-split index of each captured-or-missed frame
    final_label: np.ndarray       # -1 when no label was produced
    route: np.ndarray
    capture_slot: np.ndarray
    final_slot: np.ndarray
    soc: np.ndarray               # per slot, after the slot's energy flows
    temp_c: np.ndarray
    ceiling: np.ndarray
    harvested_j: float            # electrical, before charge losses
    consumed_j: float             # battery-side
    curtailed_j: float            # battery-side energy refused because of the ceiling or a full battery
    b_wakes: int
    dead_slots: int
    deferred: int
    reserve_coverage: Optional[float]


def simulate(ctx: RunContext, policy: Policy, frames: np.ndarray, frame_slots: np.ndarray,
             outcomes, harvest_j: np.ndarray, temp_c: np.ndarray, night: np.ndarray) -> SimResult:
    cfg = ctx.cfg
    en = ctx.energy
    n = ctx.n_slots
    spd = ctx.slots_per_day
    slot_min = cfg.solar.slot_minutes
    C = cfg.capacity_j
    floor = cfg.battery.soc_floor * C
    eta_c, eta_d = cfg.battery.eta_charge, cfg.battery.eta_discharge
    self_discharge = cfg.battery.self_discharge_per_month / (30.0 * 86400.0) * ctx.slot_seconds * C
    queue_cap = cfg.node.queue_capacity
    deadline = ctx.deadline_slots

    nf = len(frames)
    final_label = np.full(nf, -1, dtype=np.int64)
    route = np.full(nf, UNCLASSIFIED, dtype=np.int8)
    final_slot = np.full(nf, -1, dtype=np.int64)
    tri_pred, tri_conf = outcomes.pred["triage"], outcomes.conf["triage"]

    sunrise = set(int(s) for s in ctx.sunrise if s >= 0)
    sunset = set(int(s) for s in ctx.sunset if s >= 0)

    soc_trace = np.empty(n)
    ceil_trace = np.empty(n)
    E = cfg.battery.soc_init * C
    harvested = consumed = curtailed = 0.0
    b_wakes = dead = deferred = 0

    queue: Dict[int, QItem] = {}
    order = deque()                 # (enqueue slot, frame position), lazily pruned
    ptr = 0
    obs = Obs(capacity_j=C, floor_j=floor)
    policy.reset(ctx)

    def finalize_refined(pos: int, op: str, t: int, r: int):
        final_label[pos] = outcomes.pred[op][frames[pos]]
        route[pos] = r
        final_slot[pos] = t

    for t in range(n):
        h = float(harvest_j[t])
        is_night = bool(night[t])
        obs.t, obs.day = t, t // spd
        obs.hour = (t % spd) * slot_min / 60.0
        obs.night = is_night
        obs.energy_j, obs.soc = E, E / C
        obs.harvest_j, obs.temp_c = h, float(temp_c[t])
        obs.b_will_wake = False

        if t in sunset:
            ctx.forecaster.mark_sunset(t)
        if t in sunrise:
            policy.on_sunrise(obs)
        ceil = min(max(float(policy.ceiling(obs)), 0.0), 1.0)
        obs.ceiling = ceil

        sleep_b = en.sleep_j / eta_d
        budget = E - floor + eta_c * h - sleep_b
        spent = sleep_b
        alive = (E + eta_c * h - sleep_b) > 0.0
        obs.curtail_hint_j = max(0.0, E + eta_c * h - sleep_b - ceil * C)

        def spend(load_j: float) -> bool:
            nonlocal budget, spent
            b = load_j / eta_d
            if b <= budget + 1e-12:
                budget -= b
                spent += b
                return True
            return False

        # -- arrivals ------------------------------------------------------------
        essential = en.sleep_j
        now_list = []                       # (pos, op, triaged)
        while ptr < nf and frame_slots[ptr] == t:
            pos = ptr
            ptr += 1
            f = frames[pos]
            cap_j = en.capture_j(is_night)
            essential += cap_j + en.triage_j + en.store_j
            final_slot[pos] = t
            if not alive:
                route[pos] = DROPPED
                continue
            action = policy.pre_capture(obs, pos)
            if action == "drop" or not spend(cap_j):
                route[pos] = DROPPED
                continue
            if action == "direct":
                now_list.append((pos, policy.now_op(obs), False))
                continue
            if not spend(en.triage_j):
                route[pos] = UNCLASSIFIED
                continue
            p, c = int(tri_pred[f]), float(tri_conf[f])
            final_label[pos] = p
            route[pos] = TRIAGE
            g = outcomes.gain(policy.refine_op, p, c)
            decision = policy.post_triage(obs, pos, p, c, g)
            if decision == "now":
                now_list.append((pos, policy.now_op(obs), True))
            elif decision == "defer" and len(queue) < queue_cap and spend(en.store_j):
                queue[pos] = QItem(pos, t, g, p, c)
                order.append((t, pos))
                deferred += 1
                final_slot[pos] = -1        # not final yet

        # -- refinement ----------------------------------------------------------
        obs.b_will_wake = bool(now_list)
        obs.energy_j = E
        obs.queue_len = len(queue)
        while order and order[0][1] not in queue:
            order.popleft()
        obs.oldest_age_slots = (t - order[0][0]) if order else 0

        batch: List[QItem] = []
        if queue and policy.wants_batch(obs):
            batch = policy.refine_batch(obs, list(queue.values()))
            for it in batch:
                queue.pop(it.frame, None)

        if now_list or batch:
            woke = spend(en.wake_j)
            if woke:
                b_wakes += 1
            for pos, op, triaged in now_list:
                if woke and spend(en.refine_j[op]):
                    finalize_refined(pos, op, t, REFINED_NOW)
                elif not triaged:
                    f = frames[pos]
                    if spend(en.triage_j):
                        final_label[pos] = int(tri_pred[f])
                        route[pos] = TRIAGE
                    else:
                        route[pos] = UNCLASSIFIED
            for it in batch:
                if woke and spend(en.refine_j[policy.refine_op]):
                    finalize_refined(it.frame, policy.refine_op, t, REFINED_LATER)
                else:
                    queue[it.frame] = it    # could not afford it after all: back in the queue

        # -- deadline: a deferred frame keeps its triage label ------------------
        while order:
            t0, pos = order[0]
            if pos not in queue:
                order.popleft()
            elif t - t0 >= deadline:
                order.popleft()
                queue.pop(pos)
                final_slot[pos] = t
            else:
                break

        # -- energy update -------------------------------------------------------
        inflow = eta_c * h
        e_uncapped = E + inflow - spent
        if inflow - spent > 0.0:
            e_new = min(e_uncapped, max(E, ceil * C))
        else:
            e_new = e_uncapped
        e_new = min(e_new, C)
        curtailed += e_uncapped - e_new
        e_new -= self_discharge
        if e_new <= 0.0:
            e_new = 0.0
            dead += 1
        harvested += h
        consumed += spent
        E = e_new

        soc_trace[t] = E / C
        ceil_trace[t] = ceil
        ctx.forecaster.record(t, h, essential)

    for pos in list(queue):             # still waiting when the window ends
        final_slot[pos] = n - 1

    return SimResult(frames=frames, final_label=final_label, route=route,
                     capture_slot=frame_slots.copy(), final_slot=final_slot,
                     soc=soc_trace, temp_c=np.asarray(temp_c, dtype=np.float64),
                     ceiling=ceil_trace, harvested_j=harvested, consumed_j=consumed,
                     curtailed_j=curtailed, b_wakes=b_wakes, dead_slots=dead,
                     deferred=deferred, reserve_coverage=ctx.forecaster.coverage())
