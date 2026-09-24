"""The closed-loop, slot-by-slot system simulator.

Every policy in the comparison -- ours, every baseline, both oracles -- runs
through the *same* loop over the *same* realised solar trace, battery and frame
stream, so the only thing that differs between rows of the results table is the
decision rule.

Per-image correctness and confidence at every (resolution, exit) operating
point is precomputed once (scripts/02) and looked up here by the stream's
`image_index`, which keeps a multi-seed, multi-site, multi-policy sweep fast
without putting network forward passes inside the simulation loop.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import numpy as np

from ecoexit.control.pricing import (
    build_operating_points, concave_envelope, ShadowPriceController,
)
from ecoexit.control.baselines import Action
from ecoexit.sim.stream import expected_value_curve
from ecoexit.wear.rainflow import CycleLifeCurve, OnlineWearPricer


def _self_discharge_j(cfg, capacity_j: float) -> float:
    per_slot = cfg.battery.self_discharge_per_month / (30.0 * 24 * 3600) * cfg.solar.slot_seconds
    return per_slot * capacity_j


@dataclass
class LoopResult:
    soc_frac: np.ndarray
    depleted: np.ndarray
    energy_j: np.ndarray
    captured: np.ndarray        # True if duty==wake this slot
    correct: np.ndarray         # True if the inference (if any) was correct
    chosen_res: np.ndarray
    chosen_exit: np.ndarray
    lambda_trace: Optional[np.ndarray] = None
    mu_trace: Optional[np.ndarray] = None
    controller_overhead_j: float = 0.0
    online_wear: Optional[float] = None
    predicted: Optional[np.ndarray] = None   # predicted class, for macro-F1
    labels: Optional[np.ndarray] = None      # true class, for macro-F1


def _lookup(per_image, img_idx: int, res_idx: int, exit_idx: int):
    """per_image[res_idx][exit_idx] -> (correct[n] bool, conf[n] float)"""
    correct_arr, conf_arr = per_image[res_idx][exit_idx]
    return bool(correct_arr[img_idx]), float(conf_arr[img_idx])


def _horizon_slots(cfg) -> int:
    return max(int(round(cfg.control.horizon_hours * 3600.0 / cfg.solar.slot_seconds)), 1)


def run_shadow_price(cfg, energy_table, sense_energy, acc_grid, per_image,
                     harvest_j, lower_bound_j_per_day, slots_per_day, replan_slots,
                     stream, capacity_j, soc_min_j, elevation_deg,
                     kappa: float = 0.0, wear_mode: str = "path",
                     confidence_aware: bool = True,
                     use_value_estimate: bool = True) -> LoopResult:
    """The proposed controller.

    `lower_bound_j_per_day[d]` is the conformal lower bound on day d's total
    harvest. Each replan prorates that day's bound across the block's slots.

    `wear_mode` selects the wear price, and is the axis of the ablation that
    decides whether Claim 2 survives:
        "none"   -- energy price only (lambda alone)
        "proxy"  -- energy-proportional wear, the honest strawman; constant, so
                    mathematically equivalent to raising lambda
        "path"   -- the rainflow residual price, path-dependent
    """
    n = len(harvest_j)
    idle_j = cfg.energy.p_idle_w * cfg.solar.slot_seconds
    points = build_operating_points(energy_table, sense_energy, acc_grid,
                                    idle_energy_j=idle_j,
                                    wake_energy_j=cfg.energy.e_wake_j)
    hull = concave_envelope(points)

    curve = CycleLifeCurve.fit(cfg.battery.dod_points, cfg.battery.cycles_at_dod)
    pricer = OnlineWearPricer(curve) if wear_mode in ("path", "proxy") else None
    ctrl = ShadowPriceController(hull, kappa=kappa, wear_pricer=pricer,
                                 capacity_j=capacity_j)
    if wear_mode == "proxy" and pricer is not None:
        # Freeze the price at a nominal depth: this is what prior systems do,
        # and being constant it collapses into lambda by construction.
        nominal = float(getattr(cfg.wear, "proxy_nominal_depth", 0.3))
        const_mu = kappa * curve.marginal_wear(nominal) / max(capacity_j, 1e-9)
        ctrl.wear_price = lambda soc_frac, _c=const_mu: _c

    sd_j = _self_discharge_j(cfg, capacity_j)
    soc_reserve_j = soc_min_j + cfg.control.reserve_frac * capacity_j
    horizon = _horizon_slots(cfg)

    soc = np.empty(n + 1)
    soc[0] = capacity_j * cfg.battery.soc_init_frac
    depleted = np.zeros(n, dtype=bool)
    energy = np.zeros(n)
    captured = np.zeros(n, dtype=bool)
    correct = np.zeros(n, dtype=bool)
    res_choice = np.full(n, -1)
    exit_choice = np.full(n, -1)
    lam_trace = np.zeros(n)
    mu_trace = np.zeros(n)

    n_res, n_exit = acc_grid.shape
    overhead_j = 0.0
    n_days = len(lower_bound_j_per_day)
    v_hat = expected_value_curve(elevation_deg, cfg.stream) if use_value_estimate else np.ones(n)

    if pricer is not None:
        pricer.update(soc[0] / capacity_j)

    for b in range(0, n, replan_slots):
        block_end = min(b + replan_slots, n)
        n_block_slots = block_end - b
        day = min(b // slots_per_day, n_days - 1)
        budget = lower_bound_j_per_day[day] * (n_block_slots / slots_per_day)
        ctrl.replan(harvest_budget_j=budget, soc_now_j=soc[b],
                    soc_reserve_j=soc_reserve_j, soc_max_j=capacity_j,
                    n_block_slots=n_block_slots, horizon_slots=horizon,
                    value_scales=v_hat[b:block_end])
        # Planner cost scales with bisection iterations x hull size x the number
        # of value scales averaged over, which is the block length. At the
        # defaults that is ~60 x 3 x 30 comparisons per replan, roughly 30x the
        # original estimate -- worth stating rather than keeping the old number.
        overhead_j += 7.0e-5

        for t in range(b, block_end):
            soc_frac_now = soc[t] / capacity_j
            lam_trace[t] = ctrl.lam
            default = ctrl.decide(soc_frac_now, value_scale=v_hat[t])
            mu_trace[t] = ctrl.mu
            overhead_j += 3.0e-7

            r = k = None
            is_correct = False
            e = idle_j

            if default.duty == 1:
                r = default.res_idx
                img = stream.image_index[t]
                if confidence_aware:
                    thetas = ctrl.exit_thresholds(acc_grid, energy_table, r,
                                                  soc_frac_now, value_scale=v_hat[t])
                    k = 0
                    is_correct, conf = _lookup(per_image, img, r, 0)
                    while k < n_exit - 1 and conf >= thetas[k]:
                        k += 1
                        is_correct, conf = _lookup(per_image, img, r, k)
                else:
                    k = default.exit_idx
                    is_correct, _ = _lookup(per_image, img, r, k)
                e = idle_j + cfg.energy.e_wake_j + sense_energy[r] + energy_table[(r, k)]

            # A device cannot spend energy it does not have. An unaffordable
            # action browns out to idle: no inference, no value credited, only
            # the idle draw paid. Without this a reckless policy collects free
            # accuracy during blackouts it could not physically afford.
            tentative = soc[t] + cfg.battery.eta_charge * harvest_j[t] \
                - e / cfg.battery.eta_discharge - sd_j
            if default.duty == 1 and tentative < soc_min_j:
                e = idle_j
                r = k = None
                is_correct = False

            soc_next = soc[t] + cfg.battery.eta_charge * harvest_j[t] \
                - e / cfg.battery.eta_discharge - sd_j
            if soc_next < soc_min_j:
                depleted[t] = True
                soc_next = max(soc_next, 0.0)
            soc[t + 1] = min(soc_next, capacity_j)
            energy[t] = e

            if pricer is not None:
                pricer.update(soc[t + 1] / capacity_j)

            if r is not None:
                captured[t] = True
                correct[t] = is_correct
                res_choice[t] = r
                exit_choice[t] = k

    return LoopResult(soc[1:] / capacity_j, depleted, energy, captured, correct,
                      res_choice, exit_choice, lam_trace, mu_trace, overhead_j,
                      online_wear=(pricer.total_wear() if pricer is not None else None))


def run_fixed_policy(cfg, energy_table, sense_energy, per_image,
                     harvest_j, stream, capacity_j, soc_min_j,
                     act_fn: Callable[[int, float], Action]) -> LoopResult:
    """Generic driver for every policy whose decision needs only (t, soc_frac)
    plus its own internal state closed over in `act_fn`."""
    n = len(harvest_j)
    idle_j = cfg.energy.p_idle_w * cfg.solar.slot_seconds
    sd_j = _self_discharge_j(cfg, capacity_j)

    soc = np.empty(n + 1)
    soc[0] = capacity_j * cfg.battery.soc_init_frac
    depleted = np.zeros(n, dtype=bool)
    energy = np.zeros(n)
    captured = np.zeros(n, dtype=bool)
    correct = np.zeros(n, dtype=bool)
    res_choice = np.full(n, -1)
    exit_choice = np.full(n, -1)

    for t in range(n):
        a = act_fn(t, soc[t] / capacity_j)

        r = k = None
        is_correct = False
        e = idle_j
        if a.duty == 1:
            img = stream.image_index[t]
            is_correct, _ = _lookup(per_image, img, a.res_idx, a.exit_idx)
            e = idle_j + cfg.energy.e_wake_j + sense_energy[a.res_idx] \
                + energy_table[(a.res_idx, a.exit_idx)]
            r, k = a.res_idx, a.exit_idx

        tentative = soc[t] + cfg.battery.eta_charge * harvest_j[t] \
            - e / cfg.battery.eta_discharge - sd_j
        if a.duty == 1 and tentative < soc_min_j:
            e = idle_j
            r = k = None
            is_correct = False

        soc_next = soc[t] + cfg.battery.eta_charge * harvest_j[t] \
            - e / cfg.battery.eta_discharge - sd_j
        if soc_next < soc_min_j:
            depleted[t] = True
            soc_next = max(soc_next, 0.0)
        soc[t + 1] = min(soc_next, capacity_j)

        if r is not None:
            captured[t] = True
            correct[t] = is_correct
            res_choice[t] = r
            exit_choice[t] = k
        energy[t] = e

    return LoopResult(soc[1:] / capacity_j, depleted, energy, captured, correct,
                      res_choice, exit_choice)


def run_action_sequence(cfg, energy_table, sense_energy, per_image, harvest_j,
                        stream, capacity_j, soc_min_j, actions_list: List[Action],
                        action_indices: np.ndarray) -> LoopResult:
    """Replays a precomputed action sequence (an oracle's DP solution) through
    exactly the same physics and bookkeeping as every other policy, so only the
    decision rule differs."""
    def act_fn(t, soc_frac):
        return actions_list[int(action_indices[t])]
    return run_fixed_policy(cfg, energy_table, sense_energy, per_image,
                            harvest_j, stream, capacity_j, soc_min_j, act_fn)
