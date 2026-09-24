"""Solar irradiance traces.

Default path is fully offline: analytic clear-sky geometry (Haurwitz) times a
persistent stochastic clear-sky-index process. That keeps the whole repo
runnable with no network and no registration, and it gives us ground-truth
control over cloudiness so the cross-climate experiment is a real test.

`load_pvgis` swaps in a real PVGIS hourly series when a network is available.
"""
from dataclasses import dataclass
import numpy as np


SOLAR_CONSTANT_HAURWITZ = 1098.0


@dataclass
class SolarTrace:
    site: str
    slot_seconds: int
    ghi: np.ndarray            # W/m^2, per slot
    ghi_clear: np.ndarray      # W/m^2 clear-sky envelope, per slot
    elevation_deg: np.ndarray  # solar elevation, per slot
    harvest_j: np.ndarray      # J harvested per slot after panel conversion

    @property
    def n_slots(self) -> int:
        return len(self.ghi)

    @property
    def kc(self) -> np.ndarray:
        """Clear-sky index -- the part a forecaster actually has to predict."""
        out = np.zeros_like(self.ghi)
        m = self.ghi_clear > 1.0
        out[m] = self.ghi[m] / self.ghi_clear[m]
        return out


def _clear_sky(lat_deg: float, day_of_year: int, n_slots: int, slot_seconds: int):
    """Haurwitz clear-sky GHI from solar geometry. No site data required."""
    t_hours = np.arange(n_slots) * slot_seconds / 3600.0
    doy = day_of_year + (t_hours // 24).astype(int)
    hour = t_hours % 24.0

    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    omega = np.deg2rad(15.0 * (hour - 12.0))
    phi = np.deg2rad(lat_deg)

    cos_z = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(omega)
    cos_z = np.clip(cos_z, 0.0, 1.0)

    ghi = np.zeros_like(cos_z)
    day = cos_z > 0.02
    ghi[day] = SOLAR_CONSTANT_HAURWITZ * cos_z[day] * np.exp(-0.059 / cos_z[day])
    return np.maximum(ghi, 0.0), np.rad2deg(np.arcsin(cos_z))


def _cloud_process(n: int, kc_mean: float, rho: float, rng: np.random.Generator):
    """AR(1) in logit space -> persistent, bounded, skewed cloudiness.

    Persistence matters: an i.i.d. cloud process would make forecasting
    pointless and would flatter any look-ahead controller for the wrong reason.
    """
    mu = np.log(kc_mean / (1.0 - kc_mean))
    sigma = 0.85
    z = np.empty(n)
    z[0] = mu + rng.normal(0.0, sigma)
    innov = rng.normal(0.0, sigma * np.sqrt(1.0 - rho ** 2), size=n)
    for i in range(1, n):
        z[i] = mu + rho * (z[i - 1] - mu) + innov[i]
    kc = 1.0 / (1.0 + np.exp(-z))
    return np.clip(kc, 0.04, 1.0)


def make_trace(site: str, site_cfg: dict, solar_cfg, seed: int = 0,
               cache_dir: str = "./data/pvgis") -> SolarTrace:
    """Dispatch on `solar_cfg.source`: analytic by default, PVGIS when asked.

    The PVGIS path needs a network on first call only; the series is cached to
    disk afterwards.
    """
    if getattr(solar_cfg, "source", "analytic") == "pvgis":
        return make_trace_pvgis(site, site_cfg, solar_cfg, cache_dir=cache_dir)
    return make_trace_analytic(site, site_cfg, solar_cfg, seed=seed)


def make_trace_pvgis(site: str, site_cfg: dict, solar_cfg,
                     cache_dir: str = "./data/pvgis") -> SolarTrace:
    """Real measured hourly irradiance at this site's coordinates.

    The clear-sky envelope is still computed analytically, because the conformal
    forecaster predicts the clear-sky *index* and needs a denominator; PVGIS
    supplies the numerator.
    """
    from ecoexit.solar.pvgis import fetch_pvgis_hourly, hourly_to_slots

    n = int(solar_cfg.days * 24 * 3600 / solar_cfg.slot_seconds)
    hourly = fetch_pvgis_hourly(site_cfg["lat"], site_cfg["lon"],
                                solar_cfg.pvgis_year, cache_dir=cache_dir)

    start_hour = (int(site_cfg.get("day_of_year", 1)) - 1) * 24
    ghi = hourly_to_slots(hourly, solar_cfg.slot_seconds, n, start_hour=start_hour)

    ghi_clear, elev = _clear_sky(site_cfg["lat"], int(site_cfg["day_of_year"]),
                                 n, solar_cfg.slot_seconds)
    # Measured irradiance can exceed the Haurwitz envelope under cloud
    # enhancement; keep the envelope above the measurement so kc stays in [0,1].
    ghi_clear = np.maximum(ghi_clear, ghi)

    p_w = ghi * solar_cfg.panel_area_m2 * solar_cfg.panel_efficiency
    return SolarTrace(site, solar_cfg.slot_seconds, ghi, ghi_clear, elev,
                      p_w * solar_cfg.slot_seconds)


def make_trace_analytic(site: str, site_cfg: dict, solar_cfg, seed: int = 0) -> SolarTrace:
    n = int(solar_cfg.days * 24 * 3600 / solar_cfg.slot_seconds)
    rng = np.random.default_rng(seed)

    ghi_clear, elev = _clear_sky(
        site_cfg["lat"], int(site_cfg["day_of_year"]), n, solar_cfg.slot_seconds
    )
    # Cloud process runs on a 5-minute grid then upsamples: real cloud fields do
    # not decorrelate every 60 s.
    step = max(1, int(300 / solar_cfg.slot_seconds))
    kc_coarse = _cloud_process(n // step + 2, site_cfg["kc_mean"], site_cfg["kc_rho"], rng)
    kc = np.repeat(kc_coarse, step)[:n]

    ghi = ghi_clear * kc
    p_w = ghi * solar_cfg.panel_area_m2 * solar_cfg.panel_efficiency
    harvest_j = p_w * solar_cfg.slot_seconds

    return SolarTrace(site, solar_cfg.slot_seconds, ghi, ghi_clear, elev, harvest_j)


def load_pvgis(lat: float, lon: float, year: int, solar_cfg, timeout: int = 30):
    """Optional: real PVGIS hourly GHI. Requires network; open API, no signup."""
    import urllib.request, json
    url = ("https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
           f"?lat={lat}&lon={lon}&startyear={year}&endyear={year}"
           "&outputformat=json&components=0")
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    hourly = np.array([h["G(i)"] for h in data["outputs"]["hourly"]], dtype=float)
    per_hour = int(3600 / solar_cfg.slot_seconds)
    return np.repeat(hourly, per_hour)
