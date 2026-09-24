"""Real measured irradiance from PVGIS, cached to disk.

PVGIS is the JRC's open solar-radiation service: no signup, no API key, global
coverage, hourly resolution. It was verified live on 2026-09-23 returning 8,760
hourly records for the Serengeti coordinates.

Pairing these series with real camera-trap capture timestamps at the *same*
coordinates is what makes the Claim 4 benchmark: a co-located event-and-harvest
dataset, which does not currently exist and is independently citable.

The analytic Haurwitz-plus-AR(1) generator in `traces.py` stays as the offline
fallback, so the repository still runs with no network.
"""
from typing import Optional, Tuple
import json
import os
import numpy as np


PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"


def _cache_path(cache_dir: str, lat: float, lon: float, year: int) -> str:
    return os.path.join(cache_dir, f"pvgis_{lat:.4f}_{lon:.4f}_{year}.json")


def fetch_pvgis_hourly(lat: float, lon: float, year: int, cache_dir: str = "./data/pvgis",
                       timeout: int = 60, force: bool = False) -> np.ndarray:
    """Hourly global irradiance on the panel plane, W/m^2, for one calendar year.

    Cached on first fetch so a multi-seed sweep does not hammer a public
    service. Raises on failure rather than silently falling back, because a
    silent fallback to synthetic data inside a run labelled "real irradiance"
    is exactly the kind of thing that invalidates a results table.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, lat, lon, year)

    if os.path.exists(path) and not force:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        import urllib.request
        url = (f"{PVGIS_URL}?lat={lat}&lon={lon}&startyear={year}&endyear={year}"
               "&outputformat=json&components=0")
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    hourly = data.get("outputs", {}).get("hourly")
    if not hourly:
        raise ValueError(f"PVGIS returned no hourly series for ({lat}, {lon}, {year})")
    return np.array([h["G(i)"] for h in hourly], dtype=float)


def solar_elevation(lat_deg: float, n_slots: int, slot_seconds: int,
                    day_of_year_start: int = 1) -> np.ndarray:
    """Solar elevation on the slot grid, from geometry alone.

    PVGIS gives irradiance but the stream's diel weighting and the report's
    figures need elevation, so it is computed here rather than fetched.
    """
    t_hours = np.arange(n_slots) * slot_seconds / 3600.0
    doy = day_of_year_start + (t_hours // 24).astype(int)
    hour = t_hours % 24.0
    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    omega = np.deg2rad(15.0 * (hour - 12.0))
    phi = np.deg2rad(lat_deg)
    cos_z = np.clip(np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(omega),
                    -1.0, 1.0)
    return np.rad2deg(np.arcsin(cos_z))


def hourly_to_slots(hourly_w_m2: np.ndarray, slot_seconds: int, n_slots: int,
                    start_hour: int = 0, interpolate: bool = True) -> np.ndarray:
    """Resample an hourly series onto the slot grid.

    Linear interpolation rather than repeat: a 60-second grid built by
    repeating hourly values produces step discontinuities at every hour
    boundary, which show up as spurious turning points in the state-of-charge
    trace and therefore as phantom rainflow cycles in the wear accounting.
    """
    per_hour = max(int(round(3600 / slot_seconds)), 1)
    n_hours_needed = int(np.ceil(n_slots / per_hour)) + 2

    src = hourly_w_m2
    if len(src) < n_hours_needed + start_hour:
        reps = int(np.ceil((n_hours_needed + start_hour) / max(len(src), 1)))
        src = np.tile(src, reps)
    src = src[start_hour:start_hour + n_hours_needed]

    if not interpolate:
        return np.repeat(src, per_hour)[:n_slots]

    x_src = np.arange(len(src)) * per_hour
    x_dst = np.arange(n_slots)
    out = np.interp(x_dst, x_src, src)
    return np.maximum(out, 0.0)
