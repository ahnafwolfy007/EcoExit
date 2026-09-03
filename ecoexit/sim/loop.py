"""The closed-loop, slot-by-slot system simulator.

Every policy in the comparison (ours + every baseline + the oracle) is run
through the *same* loop over the *same* realised solar trace, battery, and
frame stream, so the only thing that differs between rows in the results
table is the decision rule. This is what makes the comparison meaningful.

Per-image correctness/confidence at every (resolution, exit) operating point
is precomputed once (scripts/02) and looked up here by the stream's
`cifar_index` -- this keeps a 5-site x 8-policy x 14-day sweep fast (no
network forward passes inside the simulation loop) while still measuring
real CIFAR-100 image outcomes, not a synthetic accuracy model.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

from ecoexit.control.pricing import (
    build_operating_points, concave_envelope, ShadowPriceController,
)
from ecoexit.control.baselines import Action
from ecoexit.sim.stream import expected_value_curve


def _self_discharge_j(cfg, capacity_j: float) -> float:
    """Per-slot self-discharge loss, matching ecoexit.battery.model.Battery
    so every loop in this file (and the standalone Battery class used for
    wear accounting) agrees on the same energy balance."""
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
    controller_overhead_j: float = 0.0


def _lookup(per_image, cifar_idx: int, res_idx: int, exit_idx: int):
    """per_image[res_idx][exit_idx] -> (correct[n_images] bool, conf[n_images] float)"""
    correct_arr, conf_arr = per_image[res_idx][exit_idx]
    return bool(correct_arr[cifar_idx]), float(conf_arr[cifar_idx])


def run_shadow_price(cfg, energy_table, sense_energy, acc_grid, per_image,
                      harvest_j, lower_bound_j_per_day, slots_per_day, replan_slots,
                      stream, capacity_j, soc_min_j, elevation_deg, wear_of=None,
                      confidence_aware: bool = True, use_value_estimate: bool = True) -> LoopResult:
    """lower_bound_j_per_day[d] is the ACI-calibrated lower bound on day d's
    *total* harvest (Contribution B). Each 30-minute (default) replan prorates
    that day's bound evenly across the day's remaining slots -- a receding
    daily budget rather than a full sub-day harvest shape, which keeps the
    controller simple without needing an intra-day forecast model."""
    n = len(harvest_j)
    points = build_operating_points(energy_table, sense_energy, acc_grid,
                                     idle_energy_j=cfg.energy.p_idle_w * cfg.solar.slot_seconds,
                                     wake_energy_j=cfg.energy.e_wake_j)
    hull = concave_envelope(points)
    ctrl = ShadowPriceController(hull, wear_of=wear_of)
    sd_j = _self_discharge_j(cfg, capacity_j)

    soc = np.empty(n + 1)
    soc[0] = capacity_j * cfg.battery.soc_init_frac
    depleted = np.zeros(n, dtype=bool)
    energy = np.zeros(n)
    captured = np.zeros(n, dtype=bool)
    correct = np.zeros(n, dtype=bool)
    res_choice = np.full(n, -1)
    exit_choice = np.full(n, -1)
    lam_trace = np.zeros(n)

    n_res, n_exit = acc_grid.shape
    overhead_j = 0.0
    n_days = len(lower_bound_j_per_day)
    v_hat = expected_value_curve(elevation_deg, cfg.stream) if use_value_estimate else np.ones(n)

    for b in range(0, n, replan_slots):
        block_end = min(b + replan_slots, n)
        n_block_slots = block_end - b
        day = min(b // slots_per_day, n_days - 1)
        budget = lower_bound_j_per_day[day] * (n_block_slots / slots_per_day)
        ctrl.replan(harvest_budget_j=budget,
                    soc_now_j=soc[b], soc_min_j=soc_min_j, soc_max_j=capacity_j,
                    n_slots=n_block_slots)
        overhead_j += 2.3e-6  # planner call: bisection is O(iters), a few uJ on Cortex-class MCU

        for t in range(b, block_end):
            lam_trace[t] = ctrl.lam
            default = ctrl.decide(value_scale=v_hat[t])
            overhead_j += 3.0e-7  # fast-loop decision: table lookups, sub-uJ

            if default.duty == 0:
                e = cfg.energy.p_idle_w * cfg.solar.slot_seconds
                soc_next = (soc[t] + cfg.battery.eta_charge * harvest_j[t] - e / cfg.battery.eta_discharge - sd_j)
            else:
                r = default.res_idx
                img = stream.cifar_index[t]
                if confidence_aware:
                    # walk exits with the greedy stopping rule
                    thetas = ctrl.exit_thresholds(acc_grid, energy_table, r, value_scale=v_hat[t])
                    k = 0
                    is_correct, conf = _lookup(per_image, img, r, 0)
                    while k < n_exit - 1 and conf >= thetas[k]:
                        k += 1
                        is_correct, conf = _lookup(per_image, img, r, k)
                else:
                    k = default.exit_idx
                    is_correct, _ = _lookup(per_image, img, r, k)

                e = (cfg.energy.p_idle_w * cfg.solar.slot_seconds + cfg.energy.e_wake_j
                     + sense_energy[r] + energy_table[(r, k)])
                soc_next = (soc[t] + cfg.battery.eta_charge * harvest_j[t] - e / cfg.battery.eta_discharge - sd_j)

                captured[t] = True
                correct[t] = is_correct
                res_choice[t] = r
                exit_choice[t] = k

            if soc_next < soc_min_j:
                depleted[t] = True
                soc_next = max(soc_next, 0.0)
            soc[t + 1] = min(soc_next, capacity_j)
            energy[t] = e if default.duty else e

    return LoopResult(soc[1:] / capacity_j, depleted, energy, captured, correct,
                       res_choice, exit_choice, lam_trace, overhead_j)


def run_fixed_policy(cfg, energy_table, sense_energy, per_image,
                      harvest_j, stream, capacity_j, soc_min_j,
                      act_fn: Callable[[int, float], Action]) -> LoopResult:
    """Generic driver for StaticController, ReactiveLUT, ConfidenceOnlyExit
    (via a wrapping act_fn), HarvSchedQLearning and MonotoneMDPController --
    anything whose decision needs only (t, soc_frac) plus its own internal
    state closed over in `act_fn`."""
    n = len(harvest_j)
    idle_e = cfg.energy.p_idle_w * cfg.solar.slot_seconds
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
        soc_frac = soc[t] / capacity_j
        a = act_fn(t, soc_frac)

        if a.duty == 0:
            e = idle_e
        else:
            img = stream.cifar_index[t]
            is_correct, _ = _lookup(per_image, img, a.res_idx, a.exit_idx)
            e = idle_e + cfg.energy.e_wake_j + sense_energy[a.res_idx] + energy_table[(a.res_idx, a.exit_idx)]
            captured[t] = True
            correct[t] = is_correct
            res_choice[t] = a.res_idx
            exit_choice[t] = a.exit_idx

        soc_next = (soc[t] + cfg.battery.eta_charge * harvest_j[t] - e / cfg.battery.eta_discharge - sd_j)
        if soc_next < soc_min_j:
            depleted[t] = True
            soc_next = max(soc_next, 0.0)
        soc[t + 1] = min(soc_next, capacity_j)
        energy[t] = e

    return LoopResult(soc[1:] / capacity_j, depleted, energy, captured, correct,
                       res_choice, exit_choice)


def run_action_sequence(cfg, energy_table, sense_energy, per_image, harvest_j,
                         stream, capacity_j, soc_min_j, actions_list: List[Action],
                         action_indices: np.ndarray) -> LoopResult:
    """Replays a precomputed action-index sequence (the Oracle's DP solution)
    through the exact same physics and bookkeeping as `run_fixed_policy`, so
    the oracle's reported metrics are computed identically to every other
    policy's -- only its decision rule (full foresight) differs."""
    def act_fn(t, soc_frac):
        return actions_list[int(action_indices[t])]
    return run_fixed_policy(cfg, energy_table, sense_energy, per_image,
                             harvest_j, stream, capacity_j, soc_min_j, act_fn)
