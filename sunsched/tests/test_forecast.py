import numpy as np

from sunsched.env.forecast import ReserveForecaster


def _run(alpha, conformal=True, days=400, seed=0):
    rng = np.random.default_rng(seed)
    spd = 24
    n = days * spd
    fc = ReserveForecaster(n, horizon_slots=36, alpha=alpha, gamma=0.01, window_days=60,
                           eta_charge=1.0, eta_discharge=1.0, conformal=conformal)
    for d in range(days):
        amp = rng.gamma(3.0, 40.0)                       # cloudy and sunny days
        for h in range(spd):
            t = d * spd + h
            if h == 7:
                fc.predict(d * spd + 17, t)
            if h == 17:
                fc.mark_sunset(t)
            harvest = amp * max(0.0, np.sin(np.pi * (h - 7) / 10)) if 7 <= h <= 17 else 0.0
            fc.record(t, harvest, 10.0 + rng.exponential(5.0))
    return fc


def test_conformal_coverage_close_to_target():
    fc = _run(alpha=0.1)
    cov = fc.coverage()
    assert cov is not None and 0.83 <= cov <= 0.97, f"coverage {cov} far from 0.90"


def test_point_forecast_undercovers():
    """Without the conformal bound, a point forecast misses much more often --
    the reason the charge ceiling needs the bound at all."""
    cov_point = _run(alpha=0.1, conformal=False).coverage()
    cov_conf = _run(alpha=0.1, conformal=True).coverage()
    assert cov_point < cov_conf
