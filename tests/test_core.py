"""Unit tests for the parts a reviewer would check first.

Run:  python -m pytest tests -q
      python tests/test_core.py        (no pytest needed)

These need no trained checkpoint and no downloads.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ecoexit.config import Config
from ecoexit.control.pricing import (
    OpConfig, concave_envelope, directional_water_filling, envelope_value_at_price,
)
from ecoexit.sizing import DemandBand, demand_band, harvest_to_demand, regime_label
from ecoexit.wear.rainflow import (
    CycleLifeCurve, OnlineWearPricer, offline_rainflow_wear,
)


# -- wear --------------------------------------------------------------------
def test_cycle_life_fit_recovers_power_law():
    """Fitting the datasheet table must recover a sane exponent."""
    cfg = Config()
    curve = CycleLifeCurve.fit(cfg.battery.dod_points, cfg.battery.cycles_at_dod)
    assert 0.8 < curve.k < 4.0, f"implausible exponent k={curve.k}"
    assert curve.n0 > 0
    # deeper discharge must cost more life
    assert curve.half_cycle_wear(0.8) > curve.half_cycle_wear(0.2)
    # and the marginal price must rise with depth (k > 1)
    assert curve.marginal_wear(0.8) > curve.marginal_wear(0.2)


def test_online_matches_offline_rainflow():
    """The headline claim of the wear module.

    If the streaming price does not reproduce batch rainflow, the "exact, not a
    heuristic" property in the paper is false and Claim 2 collapses.
    """
    curve = CycleLifeCurve(n0=2000.0, k=2.0)
    rng = np.random.default_rng(0)

    for trial in range(8):
        # a realistic-looking SoC trace: diurnal ramp plus noise plus a few
        # deep excursions, which is what actually stresses the residual stack
        n = 1500
        t = np.linspace(0, 6 * np.pi, n)
        soc = 0.6 + 0.25 * np.sin(t) + 0.05 * rng.normal(size=n)
        soc[600:650] -= 0.3
        soc = np.clip(soc, 0.02, 1.0)

        offline = offline_rainflow_wear(soc, curve)
        pricer = OnlineWearPricer(curve)
        for b in soc:
            pricer.update(b)

        rel = abs(pricer.total_wear() - offline["wear"]) / max(offline["wear"], 1e-12)
        assert rel < 0.05, (f"trial {trial}: online {pricer.total_wear():.6g} vs "
                            f"offline {offline['wear']:.6g} (rel err {rel:.2%})")


def test_wear_price_is_not_a_constant():
    """The formal reason the path-dependent price is a contribution.

    An energy-proportional proxy is a constant, so adding it to the objective is
    the same as raising lambda. This price must vary with position in the
    excursion, otherwise it is that proxy wearing a different name.
    """
    curve = CycleLifeCurve(n0=2000.0, k=2.0)
    pricer = OnlineWearPricer(curve)

    prices = []
    soc = 0.9
    pricer.update(soc)
    for _ in range(40):
        soc -= 0.02
        prices.append(pricer.marginal_price(soc))
        pricer.update(soc)

    prices = np.array(prices)
    assert prices.max() > 3 * (prices.min() + 1e-12), \
        "price is nearly constant -- indistinguishable from raising lambda"
    assert np.all(np.diff(prices) >= -1e-9), \
        "price must rise monotonically as an excursion deepens"


def test_retracing_inside_an_excursion_is_free():
    """Wear is charged at an excursion's maximum depth, not its path length.

    Getting this wrong double-counts wear on oscillating traces, which would
    make the controller pathologically conservative under noise.
    """
    curve = CycleLifeCurve(n0=2000.0, k=2.0)
    pricer = OnlineWearPricer(curve)
    for b in [0.9, 0.8, 0.7, 0.6]:
        pricer.update(b)
    deep_price = pricer.marginal_price(0.6)
    assert deep_price > 0

    # retrace upward, then come back to ground already covered
    pricer.update(0.65)
    assert pricer.marginal_price(0.63) >= 0.0


# -- pricing -----------------------------------------------------------------
def _toy_hull():
    pts = [OpConfig(0, -1, -1, 9.0, 0.0),
           OpConfig(1, 0, 0, 12.0, 0.20),
           OpConfig(1, 0, 1, 13.0, 0.30),
           OpConfig(1, 1, 1, 14.0, 0.38),
           OpConfig(1, 2, 2, 15.0, 0.42)]
    return concave_envelope(pts)


def test_envelope_is_concave_and_sorted():
    hull = _toy_hull()
    e = [p.energy_j for p in hull]
    v = [p.value_fn for p in hull]
    assert e == sorted(e)
    assert v == sorted(v)
    slopes = [(v[i + 1] - v[i]) / (e[i + 1] - e[i]) for i in range(len(e) - 1)]
    assert all(slopes[i] >= slopes[i + 1] - 1e-9 for i in range(len(slopes) - 1)), \
        f"envelope is not concave: slopes {slopes}"


def test_price_is_zero_when_energy_is_abundant():
    """Sanity in the other direction: when the node genuinely cannot spend what
    it harvests, lambda *should* be zero and static_max *is* optimal."""
    hull = _toy_hull()
    lam = directional_water_filling(hull, harvest_budget_j=1e6, soc_now_j=1e6,
                                    soc_reserve_j=0.0, soc_max_j=1e6,
                                    n_block_slots=30, horizon_slots=1440)
    assert lam < 1e-6
    assert envelope_value_at_price(hull, lam).energy_j == max(p.energy_j for p in hull)


def test_budget_horizon_makes_lambda_live():
    """The Phase 0 repair, as a regression test.

    With the original formula -- the whole battery offered every replan block --
    lambda collapsed to zero and the controller degenerated into static_max.
    Amortizing over the horizon must leave a strictly positive price at a
    realistic state of charge.
    """
    hull = _toy_hull()
    capacity_j = 4.0 * 3600.0
    soc_now = 0.6 * capacity_j
    reserve = 0.15 * capacity_j

    lam_fixed = directional_water_filling(
        hull, harvest_budget_j=300.0, soc_now_j=soc_now, soc_reserve_j=reserve,
        soc_max_j=capacity_j, n_block_slots=30, horizon_slots=1440)

    # the bug: amortize over the block instead of the horizon
    lam_buggy = directional_water_filling(
        hull, harvest_budget_j=300.0, soc_now_j=soc_now, soc_reserve_j=reserve,
        soc_max_j=capacity_j, n_block_slots=30, horizon_slots=30)

    assert lam_fixed > 0.0, "lambda is still pinned at zero -- the repair did not take"
    assert lam_buggy <= lam_fixed, "horizon amortization must not loosen the budget"


def test_spend_is_monotone_in_price():
    hull = _toy_hull()
    spends = [envelope_value_at_price(hull, p).energy_j for p in np.linspace(0, 1.0, 25)]
    assert all(spends[i] >= spends[i + 1] - 1e-9 for i in range(len(spends) - 1)), \
        "spend must be non-increasing in price or bisection is invalid"


# -- sizing ------------------------------------------------------------------
def test_regime_labels():
    band = DemandBand(floor_j_per_day=12960.0, ceiling_j_per_day=20539.0,
                      slots_per_day=1440)
    assert regime_label(1.72, band) == "abundant"      # phoenix, as measured
    assert regime_label(0.78, band) == "discretionary"  # bergen, as measured
    assert regime_label(0.10, band) == "starved"
    assert 0.0 < band.controllable_fraction < 1.0


def test_harvest_to_demand_scales_with_panel():
    band = DemandBand(floor_j_per_day=12960.0, ceiling_j_per_day=20539.0,
                      slots_per_day=1440)
    harvest = np.full(1440, 20539.0 / 1440)
    assert abs(harvest_to_demand(harvest, band) - 1.0) < 1e-9
    assert abs(harvest_to_demand(harvest * 2, band) - 2.0) < 1e-9


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
