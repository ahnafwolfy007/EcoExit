"""Battery state-of-charge dynamics plus cycle-life wear accounting.

Two things the original EcoExit proposal's battery model dropped and this one
restores: asymmetric charge/discharge efficiency + self-discharge (so the
energy balance is honest), and depth-of-discharge-dependent wear (so a policy
can be charged for shortening the battery's life, not just for joules).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class BatteryState:
    soc_j: np.ndarray          # state of charge per slot, Joules
    wear: np.ndarray           # cumulative wear fraction per slot (0..1+)
    depleted: np.ndarray       # bool, True where SoC hit the floor
    half_cycles: list          # (start_idx, end_idx, depth) from rainflow counting


class Battery:
    def __init__(self, cfg, capacity_j: float):
        self.cfg = cfg
        self.capacity_j = capacity_j
        self.soc_min_j = cfg.soc_min_frac * capacity_j
        self._dod_curve = np.array(cfg.dod_points)
        self._cycles_curve = np.array(cfg.cycles_at_dod)

    def cycles_to_failure(self, depth_frac: float) -> float:
        """Interpolate the datasheet DoD-vs-cycle-life curve."""
        d = np.clip(depth_frac, self._dod_curve[0], self._dod_curve[-1])
        return float(np.interp(d, self._dod_curve, self._cycles_curve))

    def simulate(self, harvest_j: np.ndarray, consume_j: np.ndarray,
                 soc0_j: float = None) -> BatteryState:
        """Advance SoC slot by slot under asymmetric efficiency + self-discharge.

        b[t+1] = clip( b[t] + eta_c*harvest - consume/eta_d - self_discharge, 0, B )
        """
        n = len(harvest_j)
        soc = np.empty(n + 1)
        soc[0] = self.capacity_j * self.cfg.soc_init_frac if soc0_j is None else soc0_j
        depleted = np.zeros(n, dtype=bool)

        sd_per_slot = self.cfg.self_discharge_per_month / (30.0 * 24 * 3600) * self.cfg_slot_seconds()

        for t in range(n):
            gain = self.cfg.eta_charge * harvest_j[t]
            loss = consume_j[t] / self.cfg.eta_discharge
            b = soc[t] + gain - loss - sd_per_slot * self.capacity_j
            if b < self.soc_min_j:
                depleted[t] = True
                b = max(b, 0.0)
            soc[t + 1] = min(b, self.capacity_j)

        wear, half_cycles = self._rainflow_wear(soc[1:] / self.capacity_j)
        return BatteryState(soc[1:], wear, depleted, half_cycles)

    def cfg_slot_seconds(self) -> int:
        # Battery module is deliberately decoupled from SolarCfg; caller sets this.
        return getattr(self, "_slot_seconds", 60)

    def set_slot_seconds(self, s: int):
        self._slot_seconds = s
        return self

    def wear_from_soc_series(self, soc_frac: np.ndarray):
        """Public entry point for computing wear from a SoC trajectory that
        was already produced elsewhere (e.g. the closed-loop simulator's own
        decision loop, which must track SoC live to make decisions and so
        cannot simply call `simulate` after the fact). Avoids re-running the
        full energy balance a second time just to get the wear number."""
        return self._rainflow_wear(soc_frac)

    # -- rainflow counting -------------------------------------------------
    def _rainflow_wear(self, soc_frac: np.ndarray):
        """ASTM-style 4-point rainflow counting on the SoC trace, converted to
        a cumulative wear fraction via the DoD-vs-cycle-life curve.

        This is what lets Contribution C say "this policy costs battery life",
        not just "this policy uses this many joules".
        """
        turning = self._turning_points(soc_frac)
        half_cycles = []
        stack = []
        for p in turning:
            stack.append(p)
            while len(stack) >= 3:
                s0, s1, s2 = stack[-3], stack[-2], stack[-1]
                d1, d2 = abs(s1 - s0), abs(s2 - s1)
                if d2 >= d1:
                    depth = d1
                    if depth > 1e-6:
                        half_cycles.append((s0, s1, depth))
                    del stack[-3:-1]
                else:
                    break
        # flush remainder as half-cycles (small residual, negligible over long traces)
        for i in range(len(stack) - 1):
            depth = abs(stack[i + 1] - stack[i])
            if depth > 1e-6:
                half_cycles.append((stack[i], stack[i + 1], depth))

        # Attribute each half-cycle's wear increment as a linear ramp across the
        # trace (exact timestamps aren't needed for the aggregate wear total we
        # report -- only final_wear, i.e. wear_curve[-1], is used downstream).
        total_wear = 0.0
        for _, _, depth in half_cycles:
            n_full = self.cycles_to_failure(depth)
            total_wear += 0.5 / max(n_full, 1.0)  # a half-cycle is 0.5 of a full cycle
        wear_curve = np.linspace(0.0, total_wear, len(soc_frac))
        return wear_curve, half_cycles

    @staticmethod
    def _turning_points(x: np.ndarray):
        if len(x) < 3:
            return list(x)
        pts = [x[0]]
        for i in range(1, len(x) - 1):
            if (x[i] - x[i - 1]) * (x[i + 1] - x[i]) < 0:
                pts.append(x[i])
        pts.append(x[-1])
        return pts


def equivalent_full_cycles(half_cycles) -> float:
    return 0.5 * sum(depth for _, _, depth in half_cycles)
