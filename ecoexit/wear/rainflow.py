"""Path-dependent battery wear, priced online from the rainflow residual stack.

This is Claim 2 of the paper. The argument in three steps:

1. Rainflow cycle counting -- the standard method for turning a state-of-charge
   trace into battery wear -- is non-Markovian. The wear attributable to the
   current excursion depends on the stack of earlier unclosed turning points,
   so it cannot be written as a per-slot cost over the state. Every prior
   energy-harvesting scheduler therefore either ignores wear or charges a proxy
   proportional to energy.

2. A proxy proportional to energy is algebraically identical to raising the
   energy price lambda, so it is not a separate control knob at all.

3. But the rainflow residual is a *stack*, and the marginal wear of the current
   slot depends only on its top element. That makes an exact marginal price
   computable in O(1) amortized time, which is what `OnlineWearPricer` does.

Cycle-life model
----------------
N_f(d) = N0 * d^(-k),  the usual power form; k ~= 2 for lithium-ion.
A half-cycle of depth d consumes 0.5 / N_f(d) = 0.5 * d^k / N0 of the battery's
life. Differentiating, the marginal wear of deepening an excursion is

    dW/dd = 0.5 * k * d^(k-1) / N0

which is the price `marginal_price` returns -- in wear-fraction per unit of
depth, where depth is a fraction of usable capacity.

The indicator
-------------
Wear accrues only when the current slot sets a *new maximum depth* for the open
excursion. Retracing ground already covered inside the same excursion is free,
because the eventual closed half-cycle is charged at its maximum depth, not at
its path length. Getting this wrong double-counts wear on oscillating traces.
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class CycleLifeCurve:
    """N_f(d) = n0 * d^(-k). Fit from a datasheet DoD-vs-cycles table, or from
    measured cells (Severson et al.) once those are available."""
    n0: float = 2000.0
    k: float = 2.0

    @classmethod
    def fit(cls, dod_points, cycles_at_dod) -> "CycleLifeCurve":
        """Log-log least squares against a datasheet curve.

        log N = log n0 - k log d, so a straight line in log-log space.
        """
        d = np.asarray(dod_points, dtype=float)
        n = np.asarray(cycles_at_dod, dtype=float)
        ok = (d > 0) & (n > 0)
        if ok.sum() < 2:
            return cls()
        slope, intercept = np.polyfit(np.log(d[ok]), np.log(n[ok]), 1)
        return cls(n0=float(np.exp(intercept)), k=float(-slope))

    def cycles_to_failure(self, depth: float) -> float:
        d = max(float(depth), 1e-6)
        return self.n0 * d ** (-self.k)

    def half_cycle_wear(self, depth: float) -> float:
        """Life fraction consumed by one half-cycle of this depth."""
        d = max(float(depth), 0.0)
        if d <= 0.0:
            return 0.0
        return 0.5 * d ** self.k / self.n0

    def marginal_wear(self, depth: float) -> float:
        """d(wear)/d(depth) at this depth -- the quantity that becomes the price."""
        d = max(float(depth), 0.0)
        if d <= 0.0:
            return 0.0
        return 0.5 * self.k * d ** (self.k - 1.0) / self.n0


class OnlineWearPricer:
    """Streaming rainflow residual stack with an O(1) amortized marginal price.

    Feed it the state-of-charge fraction once per slot with `update`. Ask it for
    the current marginal wear price with `marginal_price` *before* choosing an
    action. Total wear accumulated so far is `wear_closed + wear_open`.

    Amortized cost: each turning point is pushed and popped at most once, so the
    reduction loop is O(1) amortized per slot; `marginal_price` reads only the
    top of the stack and is O(1) worst case. This runs on a Cortex-M class MCU.
    """

    def __init__(self, curve: CycleLifeCurve, min_swing: float = 1e-4):
        self.curve = curve
        self.min_swing = min_swing        # ignore numerical dither below this
        self.stack: List[float] = []       # unclosed turning points
        self.closed_cycles: List[Tuple[float, float]] = []   # (depth, count)
        self.wear_closed = 0.0

        # Turning points are detected with exactly the same local three-point
        # test the batch reference uses, on the raw samples. An earlier version
        # used a direction-with-hysteresis rule instead; on smooth synthetic
        # traces the two agree, but on real state-of-charge traces -- which
        # dither every slot from self-discharge and the idle rail -- they select
        # very different turning-point sets and the online total diverged from
        # the batch total by more than an order of magnitude.
        self._p2 = None                    # sample t-2
        self._p1 = None                    # sample t-1, the turning-point candidate
        self._current = None               # latest sample
        self._max_depth_open = 0.0         # deepest this open excursion has gone

    # -- streaming interface ------------------------------------------------
    def update(self, soc_frac: float) -> None:
        """Advance the stack with a new state-of-charge sample. O(1) amortized."""
        b = float(soc_frac)

        if self._p1 is None:
            self._p1 = self._current = b
            self.stack.append(b)
            return
        if self._p2 is None:
            self._p2, self._p1, self._current = self._p1, b, b
            return

        a, m, c = self._p2, self._p1, b
        if (m - a) * (c - m) < 0 and max(abs(m - a), abs(c - m)) >= self.min_swing:
            self._push_turning_point(m)

        self._p2, self._p1, self._current = self._p1, b, b

        d = self._open_depth(b)
        if d > self._max_depth_open:
            self._max_depth_open = d

    def _push_turning_point(self, value: float):
        self.stack.append(value)
        self._reduce(self.stack, close=True)
        self._max_depth_open = 0.0

    def _reduce(self, stack: List[float], close: bool):
        """ASTM four-point rainflow reduction.

        Whenever the middle swing of the last three residual points is no larger
        than the swing that follows it, that middle swing is a closed full
        cycle: charge it and remove it. Each point is removed at most once,
        which is what makes the amortized cost O(1) per slot.
        """
        closed = []
        while len(stack) >= 3:
            s0, s1, s2 = stack[-3], stack[-2], stack[-1]
            d1, d2 = abs(s1 - s0), abs(s2 - s1)
            if d2 >= d1 and d1 > self.min_swing:
                closed.append(d1)
                if close:
                    self.closed_cycles.append((d1, 1.0))
                    self.wear_closed += 2.0 * self.curve.half_cycle_wear(d1)
                del stack[-3:-1]
            else:
                break
        return closed

    def _open_depth(self, soc_frac: float) -> float:
        if not self.stack:
            return 0.0
        return abs(float(soc_frac) - self.stack[-1])

    # -- the price ----------------------------------------------------------
    def marginal_price(self, soc_frac: float, discharging: bool = True) -> float:
        """Marginal wear (life fraction) per unit of additional depth, now.

        `discharging=True` asks the price of drawing the state of charge
        further away from the last turning point. This is the direction a
        compute action moves it, so it is the price the controller pays.

        Returns 0 when the move would retrace ground already covered inside the
        open excursion, because the closed half-cycle is charged at its maximum
        depth and that maximum is already committed.
        """
        d = self._open_depth(soc_frac)
        if not discharging:
            return 0.0
        if d < self._max_depth_open - 1e-12:
            return 0.0            # retracing inside the excursion is free
        return self.curve.marginal_wear(max(d, self.min_swing))

    def price_per_joule(self, soc_frac: float, capacity_j: float) -> float:
        """The same price expressed per Joule drawn, which is the unit the
        controller's energy term is already in."""
        return self.marginal_price(soc_frac) / max(capacity_j, 1e-9)

    # -- accounting ---------------------------------------------------------
    def _residual(self):
        """The residual as the batch reference would see it right now.

        The batch version appends the final sample and reduces once more before
        counting residual half-cycles, so accounting has to do the same or the
        two disagree at the tail. This runs on a copy and does not disturb the
        streaming state, and it is only called when a number is asked for --
        never in the per-slot hot path.
        """
        pts = list(self.stack)
        if self._current is not None and (not pts or pts[-1] != self._current):
            pts.append(self._current)
        extra_closed = self._reduce(pts, close=False)
        halves = [abs(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        return extra_closed, halves

    def open_wear(self) -> float:
        """Wear committed by the unclosed residual, counted as half cycles."""
        extra_closed, halves = self._residual()
        w = sum(2.0 * self.curve.half_cycle_wear(d) for d in extra_closed)
        w += sum(self.curve.half_cycle_wear(d) for d in halves)
        return w

    def total_wear(self) -> float:
        return self.wear_closed + self.open_wear()

    def equivalent_full_cycles(self) -> float:
        extra_closed, halves = self._residual()
        total = sum(depth * count for depth, count in self.closed_cycles)
        total += sum(extra_closed)
        total += 0.5 * sum(halves)
        return total


# -- offline reference, used to validate the online price --------------------
def _turning_points(x: np.ndarray, min_swing: float = 1e-4) -> List[float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return [float(v) for v in x]
    pts = [float(x[0])]
    for i in range(1, len(x) - 1):
        a, b, c = x[i - 1], x[i], x[i + 1]
        if (b - a) * (c - b) < 0 and max(abs(b - a), abs(c - b)) >= min_swing:
            pts.append(float(b))
    pts.append(float(x[-1]))
    return pts


def offline_rainflow_wear(soc_frac: np.ndarray, curve: CycleLifeCurve,
                          min_swing: float = 1e-4):
    """Batch four-point rainflow over the whole trace.

    This is the ground truth the online pricer is validated against: running
    `OnlineWearPricer` slot by slot over the same trace must reproduce this
    total to numerical tolerance. `tests/test_rainflow.py` asserts exactly that.
    """
    turning = _turning_points(soc_frac, min_swing)
    stack: List[float] = []
    full_cycles: List[float] = []
    for p in turning:
        stack.append(p)
        while len(stack) >= 3:
            s0, s1, s2 = stack[-3], stack[-2], stack[-1]
            d1, d2 = abs(s1 - s0), abs(s2 - s1)
            if d2 >= d1 and d1 > min_swing:
                full_cycles.append(d1)
                del stack[-3:-1]
            else:
                break
    residual_halves = [abs(stack[i + 1] - stack[i]) for i in range(len(stack) - 1)]

    wear = sum(2.0 * curve.half_cycle_wear(d) for d in full_cycles)
    wear += sum(curve.half_cycle_wear(d) for d in residual_halves)
    efc = sum(full_cycles) + 0.5 * sum(residual_halves)
    return dict(wear=float(wear), equivalent_full_cycles=float(efc),
                full_cycles=full_cycles, residual_halves=residual_halves)


def equivalent_full_cycles(soc_frac: np.ndarray, curve: CycleLifeCurve = None) -> float:
    curve = curve or CycleLifeCurve()
    return offline_rainflow_wear(soc_frac, curve)["equivalent_full_cycles"]
