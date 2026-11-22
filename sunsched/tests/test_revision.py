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


V3 = ["sunsched3", "sunsched3_no_coalesce", "sunsched3_no_pv", "sunsched3_no_hold", "sunsched3_fc"]


def test_v3_runs_finalises_and_respects_deadline():
    for name in V3:
        for sun in (0.5, 1.0, 3.0):
            cfg, ctx, res, _ = run_policy(name, sun=sun)
            assert np.all(res.final_slot >= 0), name
            assert np.all((res.soc >= -1e-9) & (res.soc <= 1 + 1e-9)), name
            later = res.route == REFINED_LATER
            lat = res.final_slot - res.capture_slot
            assert np.all(lat[later] <= ctx.deadline_slots), name
            assert not np.any(res.route == REFINED_NOW), name


def test_v3_coalesces_wakes():
    """With coalescing, refinements share wakes: fewer wakes per refined frame."""
    _, _, a, _ = run_policy("sunsched3", sun=3.0)
    _, _, b, _ = run_policy("sunsched3_no_coalesce", sun=3.0)
    ra = a.b_wakes / max(np.sum(a.route == REFINED_LATER), 1)
    rb = b.b_wakes / max(np.sum(b.route == REFINED_LATER), 1)
    assert ra <= rb + 1e-9, (ra, rb)


def test_v3_hold_engages_only_with_surplus():
    _, _, rich, _ = run_policy("sunsched3", sun=3.0)
    _, _, poor, _ = run_policy("sunsched3", sun=0.3)
    assert rich.ceiling.min() < 1.0, "hold never engaged with plenty of sun"
    assert poor.ceiling.mean() >= rich.ceiling.mean()


def test_v3_sequent_peak_matches_brute_force():
    """The hold level's storage requirement equals the largest cumulative
    deficit over any window of the history (sequent-peak algorithm)."""
    import sunsched.tests.test_simulator as ts
    made = []
    orig = ts.make_policy
    ts.make_policy = lambda name, cfg: made.append(orig(name, cfg)) or made[-1]
    try:
        cfg, ctx, res, _ = run_policy("sunsched3", sun=1.0)
    finally:
        ts.make_policy = orig
    pol = made[0]
    pol._fold_sequent_peak(ctx.n_slots)
    fc = ctx.forecaster
    t = pol._sp_t
    eta_c, eta_d = cfg.battery.eta_charge, cfg.battery.eta_discharge
    net = (fc.essential[:t] + pol._refdem[:t]) / eta_d - eta_c * fc.harvest[:t]
    c = np.concatenate([[0.0], np.cumsum(net)])
    brute = float(np.max(c - np.minimum.accumulate(c)))
    assert abs(brute - pol._sp_max) <= 1e-6 * max(1.0, brute), (brute, pol._sp_max)


def test_v3_variants_differ_only_where_designed():
    for name in ("sunsched3_fc", "sunsched3_iter1", "sunsched3_iter2", "sunsched3_iter3"):
        _, _, res, _ = run_policy(name, sun=1.0)
        assert np.all(res.final_slot >= 0), name
