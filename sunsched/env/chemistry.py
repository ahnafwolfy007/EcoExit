"""Battery life of one simulated year under three lithium-ion chemistries.

lmo  Xu et al. (2018), already in env/battery.py and verified against its Table I.
lfp  Naumann et al. (2018, 2020).
nmc  Schmalstieg et al. (2014).

SOURCE WARNING. The lfp and nmc constants were read from one secondary page
(wiredwhite.com BESS ageing calculator, 24 Nov 2026), which says it follows the
Naumann dissertation and the Schmalstieg paper. They have not been checked
against either primary source, so no lfp or nmc number may be published before
that. Three more things are assumptions: the open-circuit voltage curve of the
nmc cell (only three points are given in the plan), the C-rate (taken from the
cycle's own depth and duration), and that both fits extrapolate to low rates.

Fade laws of the form q = k x^p are accumulated as equivalent age, so a year of
varying stress adds k^(1/p) dx to a clock and q = clock^p.
"""
from typing import Dict

import numpy as np
from scipy.interpolate import PchipInterpolator

from sunsched.env.battery import _cycle_table, aging_summary

R_GAS = 8.314462618
DAY = 86400.0

# Plan section 5.5 gives OCV 3.58, 3.68 and 4.14 V at 30, 50 and 95 % charge.
# The end points extend the neighbouring slopes (ASSUMPTION).
_OCV_Z = np.array([0.0, 0.30, 0.50, 0.95, 1.0])
_OCV_V = np.array([3.43, 3.58, 3.68, 4.14, 4.191])
NMC_OCV = PchipInterpolator(_OCV_Z, _OCV_V, extrapolate=True)
NMC_CELL_AH = 2.15


def lfp_calendar_k(soc, temp_c):
    """Capacity loss per sqrt(second)."""
    T = np.asarray(temp_c, dtype=np.float64) + 273.15
    kT = 1.2571e-5 * np.exp(-17126.0 / R_GAS * (1.0 / T - 1.0 / 298.15))
    return kT * (2.8575 * (np.asarray(soc, dtype=np.float64) - 0.5) ** 3 + 0.60225)


def lfp_cycle_b(depth, c_rate):
    """Capacity loss per sqrt(equivalent full cycle)."""
    return (0.0630 * np.asarray(c_rate) + 0.0971) * (4.0253 * (np.asarray(depth) - 0.6) ** 3 + 1.0923) / 100.0


def nmc_calendar_alpha(soc, temp_c):
    """Capacity loss per day^0.75."""
    T = np.asarray(temp_c, dtype=np.float64) + 273.15
    a = (7.543 * NMC_OCV(np.asarray(soc, dtype=np.float64)) - 23.75) * 1e6 * np.exp(-6976.0 / T)
    return np.maximum(a, 0.0)


def nmc_cycle_beta(depth, mean_soc):
    """Capacity loss per sqrt(Ah throughput of the 2.15 Ah cell)."""
    return 0.007348 * (NMC_OCV(np.asarray(mean_soc)) - 3.667) ** 2 + 0.000760 + 0.004081 * np.asarray(depth)


def _years(fade, eol: float) -> float:
    lo, hi = 0.0, 1000.0
    if fade(hi) < eol:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fade(mid) < eol else (lo, mid)
    return 0.5 * (lo + hi)


def _cycles(soc, temp_c, dt_s):
    """(count, depth, mean soc, c-rate per half cycle) of every rainflow cycle."""
    tab = _cycle_table(soc, temp_c)
    if not len(tab):
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    n, d, m, _ = tab.T
    # Solar nodes cycle at far below 1C. The trace's mean absolute rate is used
    # for every cycle, so the rate term is nearly constant.
    rate = float(np.mean(np.abs(np.diff(soc))) * 3600.0 / dt_s) if len(soc) > 1 else 0.0
    return n, d, m, np.full(len(n), rate)


def life_lfp(soc, temp_c, dt_s, days, eol=0.20) -> Dict[str, float]:
    yrs = days / 365.0
    cal = float(np.sum(lfp_calendar_k(soc, temp_c) ** 2) * dt_s) / yrs
    n, d, _, c = _cycles(soc, temp_c, dt_s)
    cyc = float(np.sum(lfp_cycle_b(d, c) ** 2 * n * d)) / yrs
    f = lambda y: np.sqrt(cal * y) + np.sqrt(cyc * y)
    y = _years(f, eol)
    return dict(life=y, calendar_share=float(np.sqrt(cal * y) / max(f(y), 1e-15)))


def life_nmc(soc, temp_c, dt_s, days, eol=0.20) -> Dict[str, float]:
    yrs = days / 365.0
    cal = float(np.sum(nmc_calendar_alpha(soc, temp_c) ** (4.0 / 3.0)) * dt_s / DAY) / yrs
    n, d, m, _ = _cycles(soc, temp_c, dt_s)
    cyc = float(np.sum(nmc_cycle_beta(d, m) ** 2 * 2.0 * NMC_CELL_AH * n * d)) / yrs
    f = lambda y: (cal * y) ** 0.75 + np.sqrt(cyc * y)
    y = _years(f, eol)
    return dict(life=y, calendar_share=float((cal * y) ** 0.75 / max(f(y), 1e-15)))


def chemistry_lives(soc, temp_c, dt_s, days, aging_cfg, capacity_wh, names=()) -> Dict[str, float]:
    out = {}
    for c in names:
        if c == "lmo":
            out["life_lmo"] = aging_summary(soc, temp_c, dt_s, days, aging_cfg, capacity_wh)["battery_years_to_eol"]
        elif c == "lfp":
            r = life_lfp(soc, temp_c, dt_s, days, aging_cfg.eol_fade)
            out["life_lfp"], out["lfp_calendar_share"] = r["life"], r["calendar_share"]
        elif c == "nmc":
            r = life_nmc(soc, temp_c, dt_s, days, aging_cfg.eol_fade)
            out["life_nmc"], out["nmc_calendar_share"] = r["life"], r["calendar_share"]
        else:
            raise ValueError(f"unknown chemistry {c!r}")
    return out
