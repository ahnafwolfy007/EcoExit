"""Tests for the code added in the paper revision (synthetic data only)."""
import dataclasses

import numpy as np

from sunsched.config import AgingCfg, Config
from sunsched.env.battery import aging_summary, aging_variants
from sunsched.sim.simulator import REFINED_LATER, REFINED_NOW
from sunsched.tests.test_forecast import _run as run_forecaster
from sunsched.tests.test_simulator import run as run_policy

NEW = ["defer_ceiling", "batch_anytime", "ceiling_conformal", "defer_ceiling_conformal"]


def _trace(days=60, seed=1):
    rng = np.random.default_rng(seed)
    spd = 288
    t = np.arange(days * spd)
    soc = 0.55 + 0.35 * np.sin(2 * np.pi * t / spd) * rng.uniform(0.3, 1.0, days).repeat(spd)
    temp = 20 + 8 * np.sin(2 * np.pi * (t / spd - 0.25)) + rng.normal(0, 1, len(t))
    return np.clip(soc, 0.05, 1.0), temp


def test_aging_variants_reproduce_default_model():
    cfg = AgingCfg()
    soc, temp = _trace()
    ref = aging_summary(soc, temp, 300.0, 60, cfg, 10.0)["battery_years_to_eol"]
    v = aging_variants(soc, temp, 300.0, 60, cfg, k_sigmas=(cfg.k_sigma,), k_Ts=(cfg.k_T,))
    assert abs(v[f"life_ksig{cfg.k_sigma:g}"] - ref) / ref < 1e-9
    assert abs(v[f"life_kT{cfg.k_T:g}"] - ref) / ref < 1e-9


def test_aging_variants_order():
    """A battery held fuller ages faster, and more so as k_sigma grows; with
    k_sigma = 0 the state of charge stops mattering for calendar ageing."""
    cfg = AgingCfg()
    temp = np.full(288 * 30, 25.0)
    hi, lo = np.full_like(temp, 0.95), np.full_like(temp, 0.40)
    ks = (0.0, 0.52, 1.04, 2.08)
    vh = aging_variants(hi, temp, 300.0, 30, cfg, k_sigmas=ks)
    vl = aging_variants(lo, temp, 300.0, 30, cfg, k_sigmas=ks)
    ratios = [vl[f"life_ksig{k:g}"] / vh[f"life_ksig{k:g}"] for k in ks]
    assert abs(ratios[0] - 1.0) < 1e-9
    assert all(b > a for a, b in zip(ratios, ratios[1:]))
    # Constant SoC, no cycling: the ratio is exactly exp(k_sigma * delta SoC).
    assert abs(ratios[2] - np.exp(1.04 * 0.55)) < 1e-6


def test_aging_variants_off_by_default():
    soc, temp = _trace(days=5)
    assert aging_variants(soc, temp, 300.0, 5, AgingCfg()) == {}


def test_shadow_aci_matches_main_forecaster():
    """The shadow ACI at the main alpha must reproduce the main coverage, and
    the shadows must not change what the policy sees."""
    import sunsched.tests.test_forecast as tf
    from sunsched.env.forecast import ReserveForecaster
    orig = ReserveForecaster.__init__

    def patched(self, *a, **k):
        k.setdefault("shadow_alphas", (0.05, 0.1, 0.2))
        orig(self, *a, **k)
    ReserveForecaster.__init__ = patched
    try:
        fc = tf._run(alpha=0.1)
    finally:
        ReserveForecaster.__init__ = orig
    base = run_forecaster(alpha=0.1)
    assert fc.misses == base.misses
    s = fc.shadow_summary()
    assert abs(s["cov_aci_0.1"] - fc.coverage()) < 1e-12
    assert s["cov_point"] < s["cov_aci_0.1"]
    # A looser risk level asks for less reserve.
    assert s["size_aci_0.2"] <= s["size_aci_0.05"] + 1e-9
    assert s["size_split_0.2"] <= s["size_split_0.05"] + 1e-9
    for k in ("aci", "split", "empirical"):
        assert 0.0 <= s[f"cov_{k}_0.05"] <= 1.0


def test_new_policies_run_and_finalise():
    for name in NEW:
        cfg, ctx, res, _ = run_policy(name)
        assert np.all(res.final_slot >= 0), name
        assert np.all(res.final_slot >= res.capture_slot), name
        assert np.all((res.soc >= -1e-9) & (res.soc <= 1 + 1e-9)), name


def test_new_policy_signatures():
    cfg, ctx, res, _ = run_policy("defer_ceiling", sun=3.0)
    assert res.deferred > 0 and not np.any(res.route == REFINED_NOW)
    assert res.ceiling.min() < 1.0, "the ceiling never engaged"
    # batch_anytime may refine at night; lazy_defer (daylight only) never does.
    for name, may_night in (("batch_anytime", True), ("lazy_defer", False)):
        cfg, ctx, res, _ = run_policy(name, sun=3.0)
        later = res.route == REFINED_LATER
        spd = cfg.slots_per_day
        hour = (res.final_slot[later] % spd) * cfg.solar.slot_minutes / 60.0
        # Same sun as test_simulator.world: night is where elevation < 0.
        elev = 60.0 * np.sin(np.pi * (hour - 6.0) / 12.0)
        at_night = np.mean(elev < 0) if later.any() else 0.0
        if may_night:
            assert at_night > 0.0, name
        else:
            assert at_night == 0.0, name


def test_ceiling_now_unchanged_by_refactor():
    """ceiling_now's ceiling must equal the published rule, recomputed here."""
    cfg, ctx, res, _ = run_policy("ceiling_now", sun=3.0)
    assert res.ceiling.min() >= cfg.control.min_ceiling - 1e-12
    assert res.ceiling.max() <= 1.0
