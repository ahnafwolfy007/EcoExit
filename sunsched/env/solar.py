"""Solar irradiance and temperature on the simulation's slot grid.

Default source is PVGIS (EU Joint Research Centre): measured-and-modelled
hourly plane-of-array irradiance G(i) and 2 m air temperature T2m, free, no
account. Air temperature matters here because calendar ageing is Arrhenius in
battery temperature, and a sealed enclosure in the sun runs well above air
temperature exactly when it charges.

The analytic generator is an offline fallback only. A run built from it is
labelled "analytic" everywhere and must not be reported as real irradiance.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import json
import os

import numpy as np

PVGIS_ENDPOINTS = ("https://re.jrc.ec.europa.eu/api/v5_2/seriescalc",
                   "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc")
DAYS = 365
HOURS = DAYS * 24


@dataclass
class SolarYear:
    """Unscaled weather for one site-year, on the slot grid (local standard time)."""
    site: str
    year: int
    slot_minutes: int
    irradiance: np.ndarray      # W/m^2 on the panel plane
    air_temp_c: np.ndarray
    elevation_deg: np.ndarray
    source: str                 # "pvgis" or "analytic"

    @property
    def n_slots(self) -> int:
        return len(self.irradiance)


def panel_orientation(lat: float) -> Tuple[int, int]:
    """Fixed tilt near latitude, facing the equator (PVGIS aspect: 0 = south)."""
    tilt = int(round(min(abs(lat), 45.0)))
    aspect = 0 if lat >= 0 else 180
    return tilt, aspect


def fetch_pvgis(lat: float, lon: float, start: int, end: int, cache_dir: str,
                timeout: int = 120, force: bool = False) -> dict:
    """Hourly PVGIS series for [start, end], cached as JSON.

    Tries the default radiation database first and falls back to ERA5, which
    covers every site this project uses. Raises rather than silently switching
    to synthetic data.
    """
    import urllib.parse
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"pvgis_{lat:.3f}_{lon:.3f}_{start}_{end}.json")
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    tilt, aspect = panel_orientation(lat)
    base = dict(lat=lat, lon=lon, startyear=start, endyear=end, pvcalculation=0,
                angle=tilt, aspect=aspect, outputformat="json")
    errors = []
    for extra in ({}, {"raddatabase": "PVGIS-ERA5"}):
        for endpoint in PVGIS_ENDPOINTS:
            url = endpoint + "?" + urllib.parse.urlencode({**base, **extra})
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SunSched/2.0"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = json.loads(r.read().decode())
                if data.get("outputs", {}).get("hourly"):
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                    return data
                errors.append(f"{url}: no hourly output")
            except Exception as e:          # network or API error: try the next option
                errors.append(f"{url}: {e}")
    raise RuntimeError("PVGIS request failed for all endpoints:\n  " + "\n  ".join(errors))


def _hourly_local(data: dict, year: int, utc_offset: float) -> Tuple[np.ndarray, np.ndarray]:
    """One 365-day local-standard-time year of hourly (G, T2m).

    PVGIS stamps are UTC ("YYYYMMDD:HHMM"). Local hours whose UTC time falls
    outside the fetched range (the first/last few hours of the span) are
    filled from the same clock hour on the nearest available day. Feb 29 is
    dropped so every year has 365 days.
    """
    by_hour: Dict[datetime, Tuple[float, float]] = {}
    for h in data["outputs"]["hourly"]:
        t = datetime.strptime(h["time"][:11], "%Y%m%d:%H")
        by_hour[t] = (float(h.get("G(i)", 0.0)), float(h.get("T2m", np.nan)))

    G = np.zeros(HOURS)
    T = np.full(HOURS, np.nan)
    local = datetime(year, 1, 1)
    i = 0
    while i < HOURS:
        if not (local.month == 2 and local.day == 29):
            utc = local - timedelta(hours=utc_offset)
            v = by_hour.get(utc)
            if v is None:
                for k in (1, -1, 2, -2, 3, -3):
                    v = by_hour.get(utc + timedelta(days=k))
                    if v is not None:
                        break
            if v is not None:
                G[i], T[i] = v
            i += 1
        local += timedelta(hours=1)

    if np.isnan(T).all():
        T[:] = 15.0
    else:
        idx = np.arange(HOURS)
        ok = ~np.isnan(T)
        T = np.interp(idx, idx[ok], T[ok])
    return np.maximum(G, 0.0), T


def solar_elevation(lat: float, lon: float, utc_offset: float, n_slots: int,
                    slot_minutes: int) -> np.ndarray:
    """Solar elevation per slot from geometry, in local standard time."""
    t_h = (np.arange(n_slots) + 0.5) * slot_minutes / 60.0
    doy = 1 + (t_h // 24).astype(int)
    clock = t_h % 24.0
    solar_time = clock + (lon - 15.0 * utc_offset) / 15.0
    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    omega = np.deg2rad(15.0 * (solar_time - 12.0))
    phi = np.deg2rad(lat)
    s = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(omega)
    return np.rad2deg(np.arcsin(np.clip(s, -1.0, 1.0)))


def elevation_at(lat: float, lon: float, utc_offset: float, day_of_year: np.ndarray,
                 clock_hours: np.ndarray) -> np.ndarray:
    """Solar elevation at arbitrary local-standard clock times (for capture times)."""
    doy = np.asarray(day_of_year, dtype=np.float64)
    solar_time = np.asarray(clock_hours, dtype=np.float64) + (lon - 15.0 * utc_offset) / 15.0
    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    omega = np.deg2rad(15.0 * (solar_time - 12.0))
    phi = np.deg2rad(lat)
    s = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(omega)
    return np.rad2deg(np.arcsin(np.clip(s, -1.0, 1.0)))


def _to_slots(hourly: np.ndarray, slot_minutes: int) -> np.ndarray:
    """Linear interpolation from hourly values (taken at mid-hour) to slot centres.

    Repeating hourly values instead creates a step at every hour boundary,
    which shows up as spurious turning points in the state-of-charge trace and
    therefore as phantom rainflow cycles.
    """
    n = HOURS * 60 // slot_minutes
    x_src = np.arange(HOURS) * 60.0 + 30.0
    x_dst = (np.arange(n) + 0.5) * slot_minutes
    return np.interp(x_dst, x_src, hourly)


def pvgis_year(site: str, site_cfg: dict, year: int, data: dict, slot_minutes: int) -> SolarYear:
    G, T = _hourly_local(data, year, site_cfg["utc_offset"])
    n = HOURS * 60 // slot_minutes
    return SolarYear(site=site, year=year, slot_minutes=slot_minutes,
                     irradiance=np.maximum(_to_slots(G, slot_minutes), 0.0),
                     air_temp_c=_to_slots(T, slot_minutes),
                     elevation_deg=solar_elevation(site_cfg["lat"], site_cfg["lon"],
                                                   site_cfg["utc_offset"], n, slot_minutes),
                     source="pvgis")


def analytic_year(site: str, site_cfg: dict, year: int, slot_minutes: int) -> SolarYear:
    """Offline fallback: Haurwitz clear sky times a persistent cloud process,
    and a seasonal-plus-diurnal temperature curve. Deterministic per site-year."""
    import zlib
    n = HOURS * 60 // slot_minutes
    # zlib.crc32, not hash(): Python randomises str hashes per process, which
    # would give every worker process a different "deterministic" year.
    rng = np.random.default_rng(zlib.crc32(f"{site}:{year}".encode()))
    elev = solar_elevation(site_cfg["lat"], site_cfg["lon"], site_cfg["utc_offset"], n, slot_minutes)
    cosz = np.clip(np.sin(np.deg2rad(elev)), 0.0, 1.0)
    clear = np.where(cosz > 0.02, 1098.0 * cosz * np.exp(-0.059 / np.maximum(cosz, 0.02)), 0.0)

    per_hour = 60 // slot_minutes
    z = np.empty(HOURS)
    z[0] = 0.0
    for i in range(1, HOURS):
        z[i] = 0.93 * z[i - 1] + rng.normal(0.0, 0.85 * np.sqrt(1 - 0.93 ** 2))
    kc = np.clip(1.0 / (1.0 + np.exp(-(1.0 + z))), 0.05, 1.0)
    irr = clear * np.repeat(kc, per_hour)[:n]

    t_h = (np.arange(n) + 0.5) * slot_minutes / 60.0
    doy = t_h / 24.0
    hemi = 1.0 if site_cfg["lat"] >= 0 else -1.0
    mean = 28.0 - 0.35 * abs(site_cfg["lat"])
    seasonal = hemi * 8.0 * np.sin(2 * np.pi * (doy - 110) / 365.0)
    diurnal = 5.0 * np.sin(2 * np.pi * ((t_h % 24.0) - 9.0) / 24.0)
    return SolarYear(site=site, year=year, slot_minutes=slot_minutes, irradiance=irr,
                     air_temp_c=mean + seasonal + diurnal, elevation_deg=elev, source="analytic")


def save_year(sy: SolarYear, path: str):
    np.savez_compressed(path, irradiance=sy.irradiance.astype(np.float32),
                        air_temp_c=sy.air_temp_c.astype(np.float32),
                        elevation_deg=sy.elevation_deg.astype(np.float32),
                        site=sy.site, year=sy.year, slot_minutes=sy.slot_minutes,
                        source=sy.source)


def load_year(path: str) -> SolarYear:
    d = np.load(path)
    return SolarYear(site=str(d["site"]), year=int(d["year"]), slot_minutes=int(d["slot_minutes"]),
                     irradiance=d["irradiance"].astype(np.float64),
                     air_temp_c=d["air_temp_c"].astype(np.float64),
                     elevation_deg=d["elevation_deg"].astype(np.float64),
                     source=str(d["source"]))


# -- derived quantities --------------------------------------------------------
def harvest_j(irradiance: np.ndarray, area_m2: float, solar_cfg, slot_seconds: float) -> np.ndarray:
    return irradiance * area_m2 * solar_cfg.panel_efficiency * solar_cfg.mppt_efficiency * slot_seconds


def irradiation_j_per_m2(irradiance: np.ndarray, solar_cfg, slot_seconds: float) -> float:
    """Electrical energy per square metre of panel over the trace."""
    return float(harvest_j(irradiance, 1.0, solar_cfg, slot_seconds).sum())


def battery_temperature(air_c: np.ndarray, irradiance: np.ndarray, gain_c: float,
                        tau_hours: float, slot_seconds: float) -> np.ndarray:
    """Air temperature plus solar heating of the enclosure, with thermal lag."""
    target = air_c + gain_c * irradiance / 1000.0
    if tau_hours <= 0 or len(target) == 0:
        return target
    from scipy.signal import lfilter
    a = 1.0 - np.exp(-slot_seconds / (tau_hours * 3600.0))
    # y[n] = (1-a) y[n-1] + a x[n], started at steady state on the first sample.
    y, _ = lfilter([a], [1.0, -(1.0 - a)], target, zi=[(1.0 - a) * target[0]])
    return y


def dawn_dusk(elevation_deg: np.ndarray, slots_per_day: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-day slot index of sunrise and sunset (-1 when the sun never rises or
    never sets that day, as at high latitude)."""
    n_days = len(elevation_deg) // slots_per_day
    dawn = np.full(n_days, -1)
    dusk = np.full(n_days, -1)
    for d in range(n_days):
        up = np.where(elevation_deg[d * slots_per_day:(d + 1) * slots_per_day] >= 0.0)[0]
        if 0 < len(up) < slots_per_day:
            dawn[d] = d * slots_per_day + up[0]
            dusk[d] = d * slots_per_day + up[-1]
    return dawn, dusk
