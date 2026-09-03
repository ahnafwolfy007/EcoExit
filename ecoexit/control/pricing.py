"""The EcoExit shadow-price controller (Contribution A).

Two loops, exactly as specified in the report:

  * `plan_price` (slow loop, called once per planning block): builds the
    concave (energy, value) envelope over every (duty, resolution, exit)
    operating point, then runs directional water-filling against the
    conformal lower-bound harvest forecast to find the single scalar price
    `lambda` for the upcoming block.

  * `FastLoop` (fast loop, called every slot): given `lambda`, `mu` (wear
    price) and the current calibrated confidence, picks (duty, resolution)
    before capture and then walks the exits with a greedy stopping rule that
    collapses to one threshold comparison per exit.

Directional water-filling (the concave-allocation optimum for exactly this
kind of energy-harvesting utility-maximisation problem) is well known in the
energy-harvesting communications literature; what's new here is applying it
to *inference configuration* instead of transmission power.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class OpConfig:
    duty: int          # 0 = sleep, 1 = wake
    res_idx: int
    exit_idx: int
    energy_j: float     # total energy this action costs this slot
    value_fn: float     # accuracy (or accuracy proxy) achievable, in [0,1]


def build_operating_points(energy_table: Dict[Tuple[int, int], float],
                            sense_energy: Dict[int, float],
                            acc_grid: np.ndarray,
                            idle_energy_j: float,
                            wake_energy_j: float) -> List[OpConfig]:
    """acc_grid[res_idx, exit_idx] = held-out accuracy at that operating point."""
    points = [OpConfig(duty=0, res_idx=-1, exit_idx=-1, energy_j=idle_energy_j, value_fn=0.0)]
    n_res, n_exit = acc_grid.shape
    for ri in range(n_res):
        for ei in range(n_exit):
            e = idle_energy_j + wake_energy_j + sense_energy[ri] + energy_table[(ri, ei)]
            points.append(OpConfig(duty=1, res_idx=ri, exit_idx=ei, energy_j=e,
                                    value_fn=float(acc_grid[ri, ei])))
    return points


def concave_envelope(points: List[OpConfig]) -> List[OpConfig]:
    """Upper concave envelope of (energy, value) points -- the set of
    operating points (plus their time-shared combinations) that are never
    strictly dominated by a mixture of two cheaper/better points. Standard
    incremental convex-hull-upper-chain construction, sorted by energy."""
    pts = sorted(points, key=lambda p: (p.energy_j, -p.value_fn))
    # keep only points on the non-decreasing running-max value frontier first
    frontier = []
    best_val = -1.0
    for p in pts:
        if p.value_fn > best_val:
            frontier.append(p)
            best_val = p.value_fn
    # upper concave hull via a monotone-stack (Andrew's-monotone-chain, upper part)
    hull: List[OpConfig] = []
    for p in frontier:
        while len(hull) >= 2:
            p1, p2 = hull[-2], hull[-1]
            # cross product sign for concavity (energy=x, value=y)
            cross = (p2.energy_j - p1.energy_j) * (p.value_fn - p1.value_fn) - \
                    (p2.value_fn - p1.value_fn) * (p.energy_j - p1.energy_j)
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def envelope_value_at_price(hull: List[OpConfig], lam: float, mu: float = 0.0,
                             wear_of: Dict = None, value_scale: float = 1.0) -> OpConfig:
    """argmax_p [ value_scale*value(p) - lam*energy(p) - mu*wear(p) ] over the
    hull. `value_scale` is the pre-capture value estimate v_hat_t: scaling the
    hull's y-axis by a positive constant never changes which points are on
    the concave envelope, so the envelope itself is built once (unweighted)
    and v_hat_t is applied only here, at decision time."""
    wear_of = wear_of or {}
    best, best_score = hull[0], -1e18
    for p in hull:
        w = wear_of.get((p.res_idx, p.exit_idx), 0.0)
        score = value_scale * p.value_fn - lam * p.energy_j - mu * w
        if score > best_score:
            best, best_score = p, score
    return best


def directional_water_filling(hull: List[OpConfig], harvest_budget_j: float,
                                soc_now_j: float, soc_min_j: float, soc_max_j: float,
                                n_slots: int, lam_bounds=(0.0, 50.0), iters: int = 40) -> float:
    """Bisect on lambda so that the hull-optimal average energy spend over
    the block, integrated with current SoC and the (conformal-bounded)
    harvest budget, does not push SoC below the floor or (irrelevantly)
    above the ceiling. This is the discretised/bisection form of directional
    water-filling: lambda is the water level, found by bisection rather than
    a closed-form KKT solve because the operating-point set is discrete.
    """
    lo, hi = lam_bounds
    # available energy this block = current SoC headroom above floor + forecast harvest
    budget = max((soc_now_j - soc_min_j) + harvest_budget_j, 0.0)
    avg_budget_per_slot = budget / max(n_slots, 1)

    def spend_at(lam):
        p = envelope_value_at_price(hull, lam)
        return p.energy_j

    # Monotonicity: higher lambda -> the argmax favours cheaper points -> spend
    # is non-increasing in lambda. Bisect for spend(lambda) ~= avg_budget.
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if spend_at(mid) > avg_budget_per_slot:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class ShadowPriceController:
    """Owns the slow-loop price and exposes the fast-loop decision rule."""

    def __init__(self, hull: List[OpConfig], wear_of: Dict = None):
        self.hull = hull
        self.wear_of = wear_of or {}
        self.lam = 1.0
        self.mu = 0.0

    def replan(self, harvest_budget_j: float, soc_now_j: float, soc_min_j: float,
               soc_max_j: float, n_slots: int):
        self.lam = directional_water_filling(
            self.hull, harvest_budget_j, soc_now_j, soc_min_j, soc_max_j, n_slots,
        )

    def decide(self, value_scale: float = 1.0) -> OpConfig:
        """Fast-loop pre-capture decision at the current price, scaled by the
        pre-capture value estimate v_hat_t (`ecoexit.sim.stream.
        expected_value_curve`). Confidence-aware early exit is layered on top
        in `sim/loop.py`, where per-image confidence is available; this call
        gives the (duty, resolution) choice made *before* that."""
        return envelope_value_at_price(self.hull, self.lam, self.mu, self.wear_of, value_scale)

    def exit_thresholds(self, acc_grid: np.ndarray, energy_table: Dict, res_idx: int,
                         value_scale: float = 1.0) -> np.ndarray:
        """theta_k(lambda, v_hat_t): the calibrated-confidence threshold above
        which the fast loop stops at exit k instead of continuing to k+1,
        from the greedy stopping rule
            continue <=> v_hat_t * value_gain > lambda * energy_gain
        approximated via each exit's marginal accuracy gain (the
        confidence-to-accuracy map itself is applied in the simulator).
        """
        n_exit = acc_grid.shape[1]
        theta = np.zeros(n_exit)
        for k in range(n_exit - 1):
            d_acc = max(acc_grid[res_idx, k + 1] - acc_grid[res_idx, k], 0.0) * value_scale
            d_e = max(energy_table[(res_idx, k + 1)] - energy_table[(res_idx, k)], 1e-9)
            theta[k] = np.clip(1.0 - d_acc / (self.lam * d_e + d_acc + 1e-9), 0.01, 0.99)
        theta[-1] = 1.0  # last exit never continues
        return theta
