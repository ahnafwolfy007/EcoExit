"""System sizing: putting the node in a regime where the controller matters.

The measurement that motivates this module: at the repo's original panel area
of 0.008 m^2, four of five sites harvest more per day than the most expensive
possible policy could spend.

    site      harvest J/day   always-max J/day   ratio
    phoenix        35,260            20,539       1.72
    dhaka          27,592            20,539       1.34
    nairobi        25,012            20,539       1.22
    munich         22,034            20,539       1.07
    bergen         16,088            20,539       0.78

Where the ratio exceeds 1 there is no allocation problem to solve, and
`static_max` is not a degenerate baseline -- it is the correct answer. A
controller that ties it there has not failed; it has been handed a problem with
no content. Reporting five sites at one fixed panel area therefore measures
nothing, and no single panel area fixes it: phoenix needs 0.0040 m^2 and bergen
0.0087 m^2 to land in the same regime.

The fix is to stop treating the panel area as a constant and start treating the
harvest-to-demand ratio as the independent variable. `solve_panel_area` inverts
the trace generator so each site can be placed at a chosen ratio, which puts
that ratio on the x-axis of every headline figure and turns the weakest point
of the evaluation into a curve a practitioner can locate their deployment on.
"""
from dataclasses import dataclass
from typing import Dict, Tuple
import copy
import numpy as np


@dataclass
class DemandBand:
    """The discretionary energy band: what the policy can actually move."""
    floor_j_per_day: float        # always asleep -- the idle rail alone
    ceiling_j_per_day: float      # always awake at the most expensive config
    slots_per_day: int

    @property
    def discretionary_j_per_day(self) -> float:
        return self.ceiling_j_per_day - self.floor_j_per_day

    @property
    def controllable_fraction(self) -> float:
        """The single number Reviewer A's strongest objection reduces to: the
        share of the energy budget the paper's knobs actually command."""
        return self.discretionary_j_per_day / max(self.ceiling_j_per_day, 1e-9)


def demand_band(cfg, energy_table: Dict[Tuple[int, int], float],
                sense_energy: Dict[int, float], acc_grid: np.ndarray) -> DemandBand:
    """Daily energy of the cheapest and most expensive admissible policies."""
    slot_s = cfg.solar.slot_seconds
    slots_per_day = int(round(86400 / slot_s))
    idle_j = cfg.energy.p_idle_w * slot_s

    n_res, n_exit = acc_grid.shape
    wake_costs = [
        idle_j + cfg.energy.e_wake_j + sense_energy[ri] + energy_table[(ri, ei)]
        for ri in range(n_res) for ei in range(n_exit)
    ]
    return DemandBand(
        floor_j_per_day=idle_j * slots_per_day,
        ceiling_j_per_day=max(wake_costs) * slots_per_day,
        slots_per_day=slots_per_day,
    )


def harvest_per_day(harvest_j: np.ndarray, slots_per_day: int) -> float:
    n_days = max(len(harvest_j) / slots_per_day, 1e-9)
    return float(np.sum(harvest_j) / n_days)


def harvest_to_demand(harvest_j: np.ndarray, band: DemandBand) -> float:
    """The regime coordinate. Below 1.0 the node cannot afford always-max and
    the allocation problem is real; above 1.0 energy is abundant and it is not."""
    return harvest_per_day(harvest_j, band.slots_per_day) / max(band.ceiling_j_per_day, 1e-9)


def regime_label(ratio: float, band: DemandBand) -> str:
    floor_ratio = band.floor_j_per_day / max(band.ceiling_j_per_day, 1e-9)
    if ratio >= 1.0:
        return "abundant"          # static_max is optimal; nothing to control
    if ratio <= floor_ratio:
        return "starved"           # cannot even hold the idle rail; nothing helps
    return "discretionary"         # the band where adaptive control earns its keep


def solve_panel_area(site: str, site_cfg: dict, solar_cfg, band: DemandBand,
                     target_ratio: float, seed: int = 0,
                     bounds: Tuple[float, float] = (1e-4, 1.0),
                     tol: float = 1e-3, iters: int = 60) -> float:
    """Panel area that puts this site at `target_ratio` of always-max demand.

    Harvest is linear in panel area, so this is a single trace build plus a
    division -- the bisection below is kept only because `make_trace` is free to
    become nonlinear later (shading, temperature derating, MPPT efficiency).
    """
    from ecoexit.solar.traces import make_trace

    def ratio_at(area: float) -> float:
        cfg2 = copy.copy(solar_cfg)
        cfg2.panel_area_m2 = area
        tr = make_trace(site, site_cfg, cfg2, seed=seed)
        return harvest_to_demand(tr.harvest_j, band)

    lo, hi = bounds
    r_lo, r_hi = ratio_at(lo), ratio_at(hi)
    if r_lo > target_ratio:
        return lo
    if r_hi < target_ratio:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ratio_at(mid) < target_ratio:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * hi:
            break
    return 0.5 * (lo + hi)


def sizing_report(sites: Dict[str, dict], solar_cfg, band: DemandBand,
                  target_ratio: float = None, seed: int = 0) -> list:
    """One row per site: current regime, and the panel area that would move it
    to `target_ratio`. This is the table that belongs in the paper's setup
    section, because it states the regime instead of leaving it implicit."""
    from ecoexit.solar.traces import make_trace

    rows = []
    for site, scfg in sites.items():
        tr = make_trace(site, scfg, solar_cfg, seed=seed)
        ratio = harvest_to_demand(tr.harvest_j, band)
        row = dict(
            site=site,
            panel_area_m2=solar_cfg.panel_area_m2,
            harvest_j_per_day=harvest_per_day(tr.harvest_j, band.slots_per_day),
            always_max_j_per_day=band.ceiling_j_per_day,
            always_sleep_j_per_day=band.floor_j_per_day,
            harvest_to_demand=ratio,
            regime=regime_label(ratio, band),
        )
        if target_ratio is not None:
            row["panel_area_for_target_m2"] = solve_panel_area(
                site, scfg, solar_cfg, band, target_ratio, seed=seed
            )
            row["target_ratio"] = target_ratio
        rows.append(row)
    return rows
