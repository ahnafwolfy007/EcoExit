"""Battery ageing: calendar plus cycle, from state-of-charge and temperature traces.

The previous version of this project modelled cycle ageing only, and priced it
online. Two findings changed that: calendar ageing dominates for a solar node
(in our traces it ended battery life 5-30x sooner than cycling did), and
rainflow-based cycle pricing is established work (Xu et al., IEEE Trans. Power
Systems 2018). Here both mechanisms are accounted for, using the semi-empirical
model in AgingCfg, and neither is claimed as a contribution.
"""
from typing import Dict, List, Tuple

import numpy as np


def _kelvin(c):
    return np.asarray(c, dtype=np.float64) + 273.15


def stress_soc(soc, cfg) -> np.ndarray:
    return np.exp(cfg.k_sigma * (np.asarray(soc, dtype=np.float64) - cfg.sigma_ref))


def stress_temp(temp_c, cfg) -> np.ndarray:
    T = _kelvin(temp_c)
    Tr = cfg.T_ref_c + 273.15
    return np.exp(cfg.k_T * (T - Tr) * Tr / T)


def stress_depth(depth, cfg) -> np.ndarray:
    d = np.clip(np.asarray(depth, dtype=np.float64), 1e-9, 1.0)
    return 1.0 / (cfg.k_d1 * d ** cfg.k_d2 + cfg.k_d3)


def calendar_damage(soc: np.ndarray, temp_c: np.ndarray, dt_s: float, cfg) -> float:
    return float(np.sum(cfg.k_t * stress_soc(soc, cfg) * stress_temp(temp_c, cfg)) * dt_s)


def turning_points(x: np.ndarray, hysteresis: float) -> List[int]:
    """Indices of reversals, ignoring wiggles smaller than `hysteresis`."""
    n = len(x)
    if n < 2:
        return [0] if n else []
    pts = [0]
    trend, cand = 0, 0
    for i in range(1, n):
        xi = x[i]
        if trend == 0:
            if xi - x[0] >= hysteresis:
                trend, cand = 1, i
            elif x[0] - xi >= hysteresis:
                trend, cand = -1, i
        elif trend == 1:
            if xi >= x[cand]:
                cand = i
            elif x[cand] - xi >= hysteresis:
                pts.append(cand)
                trend, cand = -1, i
        else:
            if xi <= x[cand]:
                cand = i
            elif xi - x[cand] >= hysteresis:
                pts.append(cand)
                trend, cand = 1, i
    if trend != 0 and cand != pts[-1]:
        pts.append(cand)
    if pts[-1] != n - 1 and abs(x[n - 1] - x[pts[-1]]) >= hysteresis:
        pts.append(n - 1)
    return pts


def rainflow(x: np.ndarray, hysteresis: float = 1e-3) -> List[Tuple[float, int, int]]:
    """ASTM E1049-85 rainflow counting. Returns (count, i_start, i_end) with
    count 1.0 for a full cycle and 0.5 for a half cycle; the range is
    |x[i_end] - x[i_start]|."""
    pts = turning_points(np.asarray(x, dtype=np.float64), hysteresis)
    cycles = []
    stack: List[int] = []
    for p in pts:
        stack.append(p)
        while len(stack) >= 3:
            rx = abs(x[stack[-1]] - x[stack[-2]])
            ry = abs(x[stack[-2]] - x[stack[-3]])
            if rx < ry:
                break
            if len(stack) == 3:
                cycles.append((0.5, stack[0], stack[1]))
                stack.pop(0)
            else:
                cycles.append((1.0, stack[-3], stack[-2]))
                last = stack[-1]
                del stack[-3:]
                stack.append(last)
    for a, b in zip(stack[:-1], stack[1:]):
        cycles.append((0.5, a, b))
    return cycles


def cycle_damage(soc: np.ndarray, temp_c: np.ndarray, cfg, hysteresis: float = 1e-3) -> Tuple[float, float]:
    """(damage, equivalent full cycles). Each cycle is stressed by its depth and
    by the mean state of charge and temperature over the samples it spans."""
    soc = np.asarray(soc, dtype=np.float64)
    temp = np.asarray(temp_c, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(soc)])
    ct = np.concatenate([[0.0], np.cumsum(temp)])
    dmg, efc = 0.0, 0.0
    for count, a, b in rainflow(soc, hysteresis):
        lo, hi = min(a, b), max(a, b)
        depth = abs(soc[b] - soc[a])
        if depth <= 0:
            continue
        m = hi - lo + 1
        s_mean = (cs[hi + 1] - cs[lo]) / m
        t_mean = (ct[hi + 1] - ct[lo]) / m
        dmg += count * float(stress_depth(depth, cfg) * stress_soc(s_mean, cfg) * stress_temp(t_mean, cfg))
        efc += count * depth
    return dmg, efc


def capacity_fade(f: float, cfg) -> float:
    return float(1.0 - cfg.alpha_sei * np.exp(-cfg.beta_sei * f) - (1.0 - cfg.alpha_sei) * np.exp(-f))


def damage_at_end_of_life(cfg) -> float:
    lo, hi = 0.0, 5.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if capacity_fade(mid, cfg) < cfg.eol_fade:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def aging_summary(soc: np.ndarray, temp_c: np.ndarray, dt_s: float, days: float, cfg,
                  capacity_wh: float) -> Dict[str, float]:
    """Annualised ageing, extrapolated from the simulated span.

    The extrapolation assumes the simulated span is representative of the
    deployment; with a full simulated year it covers every season once.
    """
    years = max(days / 365.0, 1e-9)
    f_cal = calendar_damage(soc, temp_c, dt_s, cfg) / years
    f_cyc, efc = cycle_damage(soc, temp_c, cfg)
    f_cyc /= years
    f_year = f_cal + f_cyc
    f_eol = damage_at_end_of_life(cfg)
    life = f_eol / max(f_year, 1e-15)
    replacements = max(0.0, cfg.deployment_years / life)
    return dict(
        f_calendar_per_year=f_cal,
        f_cycle_per_year=f_cyc,
        calendar_share=f_cal / max(f_year, 1e-15),
        equiv_full_cycles_per_year=efc / years,
        battery_years_to_eol=life,
        fade_after_deployment=capacity_fade(f_year * cfg.deployment_years, cfg),
        replacements_per_deployment=replacements,
        battery_kgco2e_per_deployment=replacements * cfg.battery_kgco2e_per_kwh * capacity_wh / 1000.0,
    )
