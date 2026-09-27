import numpy as np

from sunsched.config import AgingCfg
from sunsched.env import battery as B


def test_rainflow_counts_every_reversal():
    """Every reversal contributes a half cycle, so total cycles = (turning points - 1) / 2."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=3000)) / 50.0
    x = (x - x.min()) / (x.max() - x.min())
    pts = B.turning_points(x, 1e-3)
    total = sum(c for c, _, _ in B.rainflow(x, 1e-3))
    assert abs(total - (len(pts) - 1) / 2.0) < 1e-9


def test_rainflow_simple_wave():
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    cycles = B.rainflow(x, 1e-6)
    assert abs(sum(c for c, _, _ in cycles) - 2.0) < 1e-9
    assert all(abs(abs(x[b] - x[a]) - 1.0) < 1e-9 for _, a, b in cycles)


def test_stresses_are_monotone():
    cfg = AgingCfg()
    d = np.linspace(0.05, 1.0, 20)
    assert np.all(np.diff(B.stress_depth(d, cfg)) > 0), "deeper cycles must age more"
    s = np.linspace(0.0, 1.0, 20)
    assert np.all(np.diff(B.stress_soc(s, cfg)) > 0), "higher state of charge must age more"
    t = np.linspace(0.0, 50.0, 20)
    assert np.all(np.diff(B.stress_temp(t, cfg)) > 0), "hotter must age more"


def test_capacity_fade_and_end_of_life():
    cfg = AgingCfg()
    f = np.linspace(0, 1, 50)
    fades = [B.capacity_fade(v, cfg) for v in f]
    assert np.all(np.diff(fades) > 0)
    assert abs(B.capacity_fade(B.damage_at_end_of_life(cfg), cfg) - cfg.eol_fade) < 1e-6


def test_calendar_ageing_prefers_low_charge():
    cfg = AgingCfg()
    n = 10000
    full = B.calendar_damage(np.full(n, 1.0), np.full(n, 25.0), 300.0, cfg)
    mid = B.calendar_damage(np.full(n, 0.4), np.full(n, 25.0), 300.0, cfg)
    hot = B.calendar_damage(np.full(n, 1.0), np.full(n, 40.0), 300.0, cfg)
    assert full > mid and hot > full


def test_aging_summary_is_finite():
    cfg = AgingCfg()
    t = np.arange(365 * 288)
    soc = 0.6 + 0.2 * np.sin(2 * np.pi * t / 288)
    out = B.aging_summary(soc, np.full(len(t), 25.0), 300.0, 365.0, cfg, 10.0)
    assert all(np.isfinite(v) for v in out.values())
    assert 0.0 < out["calendar_share"] < 1.0
