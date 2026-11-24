"""Eligibility gates and camera split for the SolCascade test corpora.

FIREWALL. These functions read per-image metadata of corpora that must stay
unseen until the freeze. They return counts and pass/fail only: no species
name, no label distribution over time, no image. The rules were fixed before
any of these tables was opened (SolCascade plan section 5.6, with the choices
the plan leaves open written here):
  * "active deployment days" = first to last capture of the chosen window;
  * the window is the 365 days holding the most captures of that camera;
  * more than 5,000 frames in the window: keep the first 5,000 in time order.
"""
import hashlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DAY = 86400
EMPTY = frozenset({"empty", "blank", "nothing", "no animal", "none", "false trigger"})
NON_ANIMAL = EMPTY | frozenset({"human", "person", "people", "vehicle", "car", "unknown", "unclassifiable",
                                "unidentifiable", "problem", "misfire", "start", "end", "setup", "fire",
                                "", "nan"})


def parse_times(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, errors="coerce", utc=False, format="mixed")
    bad = (t.dt.year < 1995) | (t.dt.year > 2026)
    return t.mask(bad)


def best_window(days: np.ndarray, length: int = 365) -> Tuple[int, int]:
    """[start, end) day numbers of the `length`-day window with most captures."""
    d = np.sort(days)
    hi = np.searchsorted(d, d + length, side="left")
    i = int(np.argmax(hi - np.arange(len(d))))
    return int(d[i]), int(d[i]) + length


def camera_table(df: pd.DataFrame, cap: int = 5000) -> pd.DataFrame:
    """One row per camera. df: location, t (datetime64, NaT if bad), empty, one row per image."""
    rows = []
    for loc, g in df.groupby("location", sort=True, observed=True):
        t = g["t"].dropna()
        row = dict(location=loc, images=len(g), valid_time=len(t) / len(g), frames=0, span_days=0.0,
                   active_days=0, empty=float(g["empty"].mean()))
        if len(t):
            sec = t.values.astype("datetime64[s]").astype(np.int64)
            lo, hi = best_window(sec // DAY)
            w = np.sort(sec[(sec // DAY >= lo) & (sec // DAY < hi)])[:cap]
            row.update(frames=len(w), span_days=float((w[-1] - w[0]) / DAY), active_days=len(np.unique(w // DAY)),
                       window_start=lo)
        rows.append(row)
    return pd.DataFrame(rows)


def eligible(cams: pd.DataFrame, min_frames=200, min_days=60, min_valid=0.98) -> pd.Series:
    return (cams.frames >= min_frames) & (cams.span_days >= min_days) & (cams.valid_time >= min_valid)


def split_cameras(cams: pd.DataFrame, n_eval=25, n_cal=5, salt="solcascade") -> Dict[str, list]:
    """Camera-disjoint roles, balanced over activity tertiles, ordered by SHA-256."""
    c = cams.copy()
    c["rate"] = c.frames / np.maximum(c.span_days, 1.0)
    c = c.sort_values(["rate", "location"]).reset_index(drop=True)
    c["tertile"] = (np.arange(len(c)) * 3) // max(len(c), 1)
    c["h"] = [hashlib.sha256(f"{salt}|{x}".encode()).hexdigest() for x in c.location]
    order = []                                  # round robin over tertiles, hash order inside
    groups = [g.sort_values("h").location.tolist() for _, g in c.groupby("tertile")]
    while any(groups):
        for g in groups:
            if g:
                order.append(g.pop(0))
    return dict(eval=order[:n_eval], calib=order[n_eval:n_eval + n_cal], train=order[n_eval + n_cal:])


def gates(df: pd.DataFrame, has_seq: float, url_ok: float, n_eval=25, n_cal=5, min_days=60) -> dict:
    """df: location, t, empty (bool), label (str), animal (bool); one row per image."""
    cams = camera_table(df)
    ok = eligible(cams, min_days=min_days)
    e = cams[ok]
    valid = float(df["t"].notna().mean()) if len(df) else 0.0
    midnight = float((df["t"].dt.hour.eq(0) & df["t"].dt.minute.eq(0) & df["t"].dt.second.eq(0)).mean()) if len(df) else 1.0
    roles = split_cameras(e, n_eval, n_cal) if len(e) else dict(eval=[], calib=[], train=[])
    tr = df[df.location.isin(roles["train"]) & df.animal]
    cats = int((tr.label.value_counts() >= 50).sum())
    emp = float(df.loc[df.location.isin(e.location), "empty"].mean()) if len(e) else float("nan")
    bursts = float("nan")
    d = df.dropna(subset=["t"]).sort_values(["location", "t"])
    if len(d) > 1:
        gap = d.groupby("location", observed=True)["t"].diff().dt.total_seconds()
        bursts = float((gap <= 60).mean())
    out = dict(images=len(df), locations=len(cams), eligible_cameras=int(ok.sum()), train_cameras=len(roles["train"]),
               valid_time=valid, exactly_midnight=midnight, empty_share=emp, train_categories_50=cats,
               url_ok=url_ok, seq_id_share=has_seq, gap_le_60s_share=bursts,
               g1_time=valid >= 0.98 and midnight < 0.02,
               g2_cameras=len(cams) >= 45 and int(ok.sum()) >= n_eval,
               g2_with_train=len(roles["train"]) >= 10,
               g3_empty=bool(0.10 <= emp <= 0.85), g4_species=cats >= 10, g6_urls=bool(url_ok >= 0.99),
               g7_bursts=bool(has_seq >= 0.95 or bursts >= 0.2))
    out["all_pass"] = all(v for k, v in out.items() if k.startswith("g") and "_" in k and isinstance(v, bool))
    return out, cams, roles
