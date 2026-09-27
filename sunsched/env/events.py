"""Real camera captures, placed on the simulation's slot grid.

Each capture keeps its own day of year and clock time, so a location's real
diel rhythm (mostly nocturnal in CCT20) and its seasonal activity are
preserved. Captures from different calendar years at the same camera are laid
onto one simulated year; that keeps every real event while losing only the
year-to-year alignment of animal activity with a specific year's weather.

Timestamps are the cameras' own clocks. Whether a camera observed daylight
saving time is not recorded, so clock time can be off from local standard time
by up to an hour in summer; this is stated rather than corrected.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

DAYS = 365


@dataclass
class LocationStream:
    location: str
    frame_idx: np.ndarray     # indices into the evaluation-split arrays, time ordered
    slot: np.ndarray          # slot of each capture within the simulated window
    day_offset: int           # first simulated day of the window within the year
    n_days: int


def slot_in_year(timestamps: Sequence, slot_minutes: int) -> np.ndarray:
    out = np.empty(len(timestamps), dtype=np.int64)
    spd = 1440 // slot_minutes
    for i, ts in enumerate(timestamps):
        doy = ts.timetuple().tm_yday
        if ts.month > 2 and ts.year % 4 == 0 and (ts.year % 100 != 0 or ts.year % 400 == 0):
            doy -= 1                          # drop Feb 29 so every year has 365 days
        doy = min(doy, DAYS)
        out[i] = (doy - 1) * spd + (ts.hour * 60 + ts.minute) // slot_minutes
    return out


def densest_window(slots: np.ndarray, slots_per_day: int, days: int) -> int:
    """First day of the `days`-long window holding the most captures.

    Used only when simulating less than a full year (the quick preset): a real
    camera can sit idle for months, and a window drawn from a dead stretch has
    nothing to schedule.
    """
    if days >= DAYS:
        return 0
    per_day = np.bincount(slots // slots_per_day, minlength=DAYS)[:DAYS]
    window = np.convolve(per_day, np.ones(days, dtype=int), mode="valid")
    return int(np.argmax(window))


def build_streams(locations: np.ndarray, slots_in_year: np.ndarray, slots_per_day: int,
                  days: int = DAYS) -> Dict[str, LocationStream]:
    streams = {}
    for loc in sorted(set(locations.tolist())):
        idx = np.where(locations == loc)[0]
        s = slots_in_year[idx]
        start = densest_window(s, slots_per_day, days)
        lo, hi = start * slots_per_day, (start + days) * slots_per_day
        keep = (s >= lo) & (s < hi)
        order = np.argsort(s[keep], kind="stable")
        streams[str(loc)] = LocationStream(location=str(loc), frame_idx=idx[keep][order],
                                           slot=(s[keep][order] - lo), day_offset=start,
                                           n_days=days)
    return streams


def diel_profile(hours: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """Share of captures per clock hour."""
    h = np.bincount(np.asarray(hours, dtype=int) % 24, weights=weights, minlength=24)
    return h / max(h.sum(), 1e-12)
