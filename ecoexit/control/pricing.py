"""The EcoExit shadow-price controller.

Two loops:

  * `plan_price` (slow loop, once per planning block): builds the concave
    (energy, value) envelope over every (duty, resolution, exit) operating
    point, then runs directional water-filling against the conformal
    lower-bound harvest forecast to find the scalar energy price `lambda` for
    the upcoming block.

  * `FastLoop` (every slot): given `lambda`, the path-dependent wear price
    `mu_t` from the rainflow residual stack, and the current calibrated
    confidence, picks (duty, resolution) before capture and then walks the
    exits with a greedy stopping rule that collapses to one threshold
    comparison per exit.

Directional water-filling is standard in the energy-harvesting communications
literature; the contribution here is not the water-filling but the second
price, `mu_t`, which is path-dependent and therefore not expressible as a
larger `lambda` (see `ecoexit.wear.rainflow`).

BUDGET HORIZON
--------------
The first version of this file computed the block budget as

    budget = (soc_now - soc_min) + harvest_forecast_for_block

which offers the controller *the entire battery* every replan block. On the
Dhaka trace that was 7,920 J of headroom spread over a 30-slot block against a
maximum action cost of 14.3 J -- a 20x oversupply, so bisection returned
lambda = 0 at every replan, the envelope argmax collapsed to the most expensive
operating point, and the controller degenerated into `static_max`. Every
downstream ablation that depended on a live price returned bit-identical
numbers as a result.

The repair is to amortize the headroom over the horizon to the next reliable
recharge rather than over the replan block, and to amortize it down to a
*reserve* level rather than to the hard floor. A block may spend its share of
the headroom plus the harvest it expects to receive, and no more.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
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
    """Upper concave envelope of (energy, value): the operating points that are
    never dominated by a time-shared mixture of two others. Andrew's monotone
    chain, upper hull, sorted by energy."""
    pts = sorted(points, key=lambda p: (p.energy_j, -p.value_fn))
    frontier = []
    best_val = -1.0
    for p in pts:
        if p.value_fn > best_val:
            frontier.append(p)
            best_val = p.value_fn
    hull: List[OpConfig] = []
    for p in frontier:
        while len(hull) >= 2:
            p1, p2 = hull[-2], hull[-1]
            cross = (p2.energy_j - p1.energy_j) * (p.value_fn - p1.value_fn) - \
                    (p2.value_fn - p1.value_fn) * (p.energy_j - p1.energy_j)
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def envelope_value_at_price(hull: List[OpConfig], price: float,
                            value_scale: float = 1.0) -> OpConfig:
    """argmax_p [ value_scale * value(p) - price * energy(p) ] over the hull.

    `price` is the *total* marginal cost of a Joule: lambda from water-filling
    plus the wear term kappa * mu_t. Both are in the same units, which is the
    point of expressing the objective in carbon rather than energy.

    Scaling the hull's y-axis by a positive constant never changes which points
    lie on the concave envelope, so the envelope is built once, unweighted, and
    the pre-capture value estimate v_hat_t is applied only here.
    """
    best, best_score = hull[0], -1e18
    for p in hull:
        score = value_scale * p.value_fn - price * p.energy_j
        if score > best_score:
            best, best_score = p, score
    return best


def _price_bracket(hull: List[OpConfig], max_value_scale: float = 1.0) -> Tuple[float, float]:
    """A lambda range guaranteed to span every switch point on this hull.

    The largest interesting price is the steepest segment slope times the
    largest value scale in play; above it even the cheapest upgrade is refused.
    Deriving the bracket from the hull rather than hard-coding (0, 50) keeps
    bisection well-conditioned when the energy table is rescaled, which is
    exactly what `system_scale_factor` does.
    """
    slopes = []
    for a, b in zip(hull, hull[1:]):
        de = b.energy_j - a.energy_j
        if de > 1e-12:
            slopes.append((b.value_fn - a.value_fn) / de)
    if not slopes:
        return 0.0, 1.0
    return 0.0, max(slopes) * max(max_value_scale, 1e-9) * 2.0 + 1e-12


def directional_water_filling(hull: List[OpConfig], harvest_budget_j: float,
                              soc_now_j: float, soc_reserve_j: float,
                              soc_max_j: float, n_block_slots: int,
                              horizon_slots: int, value_scales=None,
                              iters: int = 60) -> float:
    """Bisect for the price at which expected spend matches the sustainable
    allowance.

    allowance_per_slot = headroom_above_reserve / horizon_slots
                       + forecast_harvest_for_block / n_block_slots

    The first term is the repair described in this module's docstring: stored
    energy is amortized over the horizon to the next reliable recharge, so a
    block is offered its *share* of the battery rather than all of it.

    `value_scales` is the sequence of pre-capture value estimates v_hat_t the
    fast loop will actually apply during this block. It matters: the fast loop
    maximizes v_hat_t * value - price * energy, so a price calibrated against
    an unweighted hull is calibrated for the wrong objective. With v_hat around
    0.15, the unweighted break-even price is roughly seven times the real one,
    and a planner that ignores this returns a price high enough to put the node
    to sleep permanently -- the mirror image of the lambda = 0 bug, and just as
    quiet.
    """
    headroom = max(soc_now_j - soc_reserve_j, 0.0)
    horizon_slots = max(int(horizon_slots), 1)
    n_block_slots = max(int(n_block_slots), 1)

    allowance_per_slot = (headroom / horizon_slots
                          + max(harvest_budget_j, 0.0) / n_block_slots)

    if value_scales is None:
        scales = np.ones(1)
    else:
        scales = np.atleast_1d(np.asarray(value_scales, dtype=float))
        if scales.size == 0:
            scales = np.ones(1)

    def expected_spend(price: float) -> float:
        return float(np.mean([envelope_value_at_price(hull, price, vs).energy_j
                              for vs in scales]))

    lo, hi = _price_bracket(hull, float(scales.max()))
    # Spend is non-increasing in price, so plain bisection converges.
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if expected_spend(mid) > allowance_per_slot:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class ShadowPriceController:
    """Owns the slow-loop price and exposes the fast-loop decision rule."""

    def __init__(self, hull: List[OpConfig], kappa: float = 0.0,
                 wear_pricer=None, capacity_j: float = 1.0):
        self.hull = hull
        self.lam = 0.0
        self.kappa = kappa               # converts wear fraction into value units
        self.wear_pricer = wear_pricer   # ecoexit.wear.rainflow.OnlineWearPricer
        self.capacity_j = capacity_j
        self.mu = 0.0                    # last wear price per Joule, for tracing

    def replan(self, harvest_budget_j: float, soc_now_j: float, soc_reserve_j: float,
               soc_max_j: float, n_block_slots: int, horizon_slots: int,
               value_scales=None):
        self.lam = directional_water_filling(
            self.hull, harvest_budget_j, soc_now_j, soc_reserve_j, soc_max_j,
            n_block_slots, horizon_slots, value_scales=value_scales,
        )

    def wear_price(self, soc_frac: float) -> float:
        """kappa * mu_t, in the same per-Joule units as lambda.

        Zero when `wear_pricer` is None (the energy-only ablation arm) and when
        the current move retraces ground already covered inside the open
        excursion, which is free under rainflow.
        """
        if self.wear_pricer is None or self.kappa <= 0.0:
            return 0.0
        return self.kappa * self.wear_pricer.price_per_joule(soc_frac, self.capacity_j)

    def total_price(self, soc_frac: float) -> float:
        self.mu = self.wear_price(soc_frac)
        return self.lam + self.mu

    def decide(self, soc_frac: float, value_scale: float = 1.0) -> OpConfig:
        """Fast-loop pre-capture (duty, resolution) choice at the current total
        price. Confidence-aware early exit is layered on top in `sim/loop.py`,
        where per-image confidence is available."""
        return envelope_value_at_price(self.hull, self.total_price(soc_frac), value_scale)

    def exit_thresholds(self, acc_grid: np.ndarray, energy_table: Dict, res_idx: int,
                        soc_frac: float, value_scale: float = 1.0) -> np.ndarray:
        """theta_k: the calibrated-confidence level above which the fast loop
        stops at exit k rather than continuing to k+1, from the greedy rule

            continue  <=>  v_hat_t * value_gain > price * energy_gain

        approximated via each exit's marginal accuracy gain. The
        confidence-to-accuracy map itself is applied in the simulator.
        """
        price = self.total_price(soc_frac)
        n_exit = acc_grid.shape[1]
        theta = np.zeros(n_exit)
        for k in range(n_exit - 1):
            d_acc = max(acc_grid[res_idx, k + 1] - acc_grid[res_idx, k], 0.0) * value_scale
            d_e = max(energy_table[(res_idx, k + 1)] - energy_table[(res_idx, k)], 1e-9)
            theta[k] = np.clip(1.0 - d_acc / (price * d_e + d_acc + 1e-9), 0.01, 0.99)
        theta[-1] = 1.0  # last exit never continues
        return theta


def energy_proportional_wear_price(kappa: float, nominal_depth: float,
                                   curve, capacity_j: float) -> float:
    """The honest strawman for the wear-price ablation.

    This is what every prior system does: charge wear in proportion to energy
    drawn, at a wear rate evaluated once at a nominal depth. It is a constant,
    so adding it to lambda is the same as raising lambda -- which is precisely
    the claim the ablation has to test. If the path-dependent price does not
    beat this arm, Claim 2 is dead.
    """
    return kappa * curve.marginal_wear(nominal_depth) / max(capacity_j, 1e-9)


def carbon_kappa(carbon_cfg, battery_capacity_wh: float,
                 value_per_kgco2e: Optional[float] = None) -> float:
    """Convert one unit of battery wear fraction into the controller's value
    units, so lambda and mu are commensurable.

    Wear of 1.0 consumes one battery, whose embodied carbon is
    `battery_kgco2e_per_kwh * capacity_kwh`. Expressing the objective in carbon
    is what makes Claim 1 operational rather than rhetorical: the controller
    literally maximizes value per kgCO2e rather than value per Joule.
    """
    battery_kg = carbon_cfg.battery_kgco2e_per_kwh * (battery_capacity_wh / 1000.0)
    if value_per_kgco2e is None:
        return battery_kg
    return battery_kg * value_per_kgco2e
