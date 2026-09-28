#!/usr/bin/env python
"""Weather traces per site-year, capture streams per camera, and the motivating measurement.

    python pipeline/03_build_environment.py                 # PVGIS (network on first run, then cached)
    python pipeline/03_build_environment.py --solar analytic   # offline fallback; label results accordingly

Writes outputs/artifacts/solar/<site>_<year>.npz, outputs/artifacts/events.npz,
and three tables: diel_mismatch.csv (when animals arrive vs when energy
arrives), sizing.csv (panel area per camera and regime), and a summary JSON.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

import numpy as np

from sunsched.cli import banner, base_parser, load_cfg, write_csv, write_json
from sunsched.config import SITES, pvgis_range
from sunsched.data import cct20
from sunsched.env import solar as S
from sunsched.env.events import build_streams, diel_profile, slot_in_year
from sunsched.experiment import always_now_demand_j
from sunsched.sim.node import build_node_energy
from sunsched.sim.outcomes import load_outcomes


def build_solar(cfg) -> dict:
    out_dir = f"{cfg.artifacts_dir}/solar"
    os.makedirs(out_dir, exist_ok=True)
    start, end = pvgis_range()
    years = {}
    for site, s in SITES.items():
        if cfg.solar.source == "pvgis":
            try:
                data = S.fetch_pvgis(s["lat"], s["lon"], start, end, cfg.data.pvgis_dir)
            except Exception as e:
                raise SystemExit(f"PVGIS unavailable for {site}: {e}\n"
                                 f"  Fix the network and re-run, or run with --solar analytic and\n"
                                 f"  report the results as synthetic-irradiance results.")
        for y in cfg.solar.years:
            sy = (S.pvgis_year(site, s, y, data, cfg.solar.slot_minutes)
                  if cfg.solar.source == "pvgis" else S.analytic_year(site, s, y, cfg.solar.slot_minutes))
            S.save_year(sy, f"{out_dir}/{site}_{y}.npz")
            years[(site, y)] = sy
        e = [S.irradiation_j_per_m2(years[(site, y)].irradiance, cfg.solar, cfg.slot_seconds)
             for y in cfg.solar.years]
        t = [years[(site, y)].air_temp_c.mean() for y in cfg.solar.years]
        print(f"  {site:<11} {cfg.solar.source}: {np.mean(e) / 3.6e6 / 365:6.3f} kWh/m^2/day "
              f"(electrical), air {np.mean(t):5.1f} C mean, years {list(cfg.solar.years)}")
    return years


def diel_mismatch(cfg, years):
    """How much of the animal activity happens when there is no sun."""
    ann = cfg.data.annotations_dir
    recs = [r for s in cct20.SPLITS for r in cct20.load_split(ann, s) if r.label != cfg.task.empty_class]
    site = SITES[cfg.experiment.main_site]
    doy = np.array([r.timestamp.timetuple().tm_yday for r in recs])
    clock = np.array([r.timestamp.hour + r.timestamp.minute / 60.0 for r in recs])
    elev = S.elevation_at(site["lat"], site["lon"], site["utc_offset"], doy, clock)

    spd = cfg.slots_per_day
    ev_share = diel_profile(clock.astype(int))
    energy = np.zeros(24)
    irr_all, elev_all = [], []
    for y in cfg.solar.years:
        sy = years[(cfg.experiment.main_site, y)]
        hours = ((np.arange(sy.n_slots) % spd) * cfg.solar.slot_minutes // 60).astype(int)
        energy += np.bincount(hours, weights=sy.irradiance, minlength=24)
        irr_all.append(sy.irradiance)
        elev_all.append(sy.elevation_deg)
    energy_share = energy / energy.sum()
    irr_all, elev_all = np.concatenate(irr_all), np.concatenate(elev_all)

    rows = [dict(hour=h, event_share=round(float(ev_share[h]), 5),
                 energy_share=round(float(energy_share[h]), 5)) for h in range(24)]
    write_csv(f"{cfg.results_dir}/tables/diel_mismatch.csv", rows)

    def energy_in(lo, hi):
        m = (elev_all >= lo) & (elev_all < hi)
        return float(irr_all[m].sum() / irr_all.sum())

    summary = dict(
        n_animal_captures=len(recs), site=cfg.experiment.main_site,
        events_sun_below_horizon=float(np.mean(elev < 0)),
        events_sun_below_15deg=float(np.mean(elev < 15)),
        events_sun_above_40deg=float(np.mean(elev >= 40)),
        energy_sun_below_15deg=energy_in(-90, 15),
        energy_sun_above_40deg=energy_in(40, 91),
        pearson_hourly_events_vs_energy=float(np.corrcoef(ev_share, energy_share)[0, 1]),
        note="capture clock times are camera clocks; DST handling is unknown (up to 1 h in summer)",
    )
    print(f"  {len(recs):,} animal captures across all 20 CCT20 cameras")
    print(f"  {summary['events_sun_below_horizon'] * 100:5.1f}% arrive with the sun below the horizon "
          f"(0% of the energy)")
    print(f"  {summary['events_sun_above_40deg'] * 100:5.1f}% arrive with the sun above 40 deg, "
          f"which delivers {summary['energy_sun_above_40deg'] * 100:.1f}% of the energy")
    return summary


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg = load_cfg(args)

    banner("1. weather")
    years = build_solar(cfg)

    banner("2. capture streams")
    oc = load_outcomes(f"{cfg.artifacts_dir}/outcomes.npz")
    d = np.load(f"{cfg.artifacts_dir}/outcomes.npz")
    ts = [datetime.fromisoformat(str(t)) for t in d["eval_timestamps"]]
    slots = slot_in_year(ts, cfg.solar.slot_minutes)
    np.savez_compressed(f"{cfg.artifacts_dir}/events.npz", eval_slot_in_year=slots,
                        eval_locations=oc.locations)
    streams = build_streams(oc.locations, slots, cfg.slots_per_day, cfg.experiment.days)
    for loc, st in sorted(streams.items(), key=lambda kv: -len(kv[1].frame_idx)):
        print(f"  camera {loc:>4}: {len(st.frame_idx):5d} captures in the simulated "
              f"{st.n_days}-day window (from day {st.day_offset})")

    banner("3. when animals arrive vs when energy arrives")
    summary = diel_mismatch(cfg, years)

    banner("4. panel sizing by regime")
    energy = build_node_energy(cfg.node, oc.macs, cfg.slot_seconds)
    spd = cfg.slots_per_day
    site = cfg.experiment.main_site
    rows = []
    for loc, st in sorted(streams.items()):
        n = st.n_days * spd
        window = slice(st.day_offset * spd, st.day_offset * spd + n)
        night = years[(site, cfg.solar.years[0])].elevation_deg[window] < 0
        demand = always_now_demand_j(st.slot, night, n, energy)
        per_m2 = np.mean([S.irradiation_j_per_m2(years[(site, y)].irradiance[window], cfg.solar,
                                                 cfg.slot_seconds) for y in cfg.solar.years])
        row = dict(location=loc, captures=len(st.frame_idx),
                   captures_per_day=round(len(st.frame_idx) / st.n_days, 2),
                   always_now_kj_per_day=round(demand / 1e3 / st.n_days, 3))
        for r in cfg.experiment.ratio_sweep:
            row[f"panel_cm2_at_ratio_{r}"] = round(r * demand / per_m2 * 1e4, 2)
        rows.append(row)
        print(f"  camera {loc:>4}: {row['captures_per_day']:6.2f} captures/day, always-now needs "
              f"{row['always_now_kj_per_day']:.2f} kJ/day")
    write_csv(f"{cfg.results_dir}/tables/sizing.csv", rows)

    # Days of autonomy: capacity over a day of essential plus inference load.
    # This decides whether the experiment can test its own hypothesis at all.
    # When it is large, a night's reserve is a rounding error against the store,
    # the charge ceiling clips to control.min_ceiling, the cell ages by calendar
    # rather than by cycling, and no policy that schedules within a day can
    # differ from simply charging to that floor.
    per_day = [r["always_now_kj_per_day"] * 1e3 for r in rows]
    mean_day_j = sum(per_day) / max(len(per_day), 1)
    autonomy = cfg.battery.capacity_wh * 3600.0 / max(mean_day_j, 1e-9)
    summary["battery_capacity_wh"] = cfg.battery.capacity_wh
    summary["mean_daily_demand_j"] = round(mean_day_j, 1)
    summary["days_of_autonomy"] = round(autonomy, 1)
    print(f"\n  battery {cfg.battery.capacity_wh:.1f} Wh ({cfg.battery.capacity_wh * 3600:.0f} J)"
          f" vs {mean_day_j:.0f} J/day -> {autonomy:.0f} days of autonomy")
    if autonomy > 30:
        print("  WARNING: that is far more store than a night needs, so the conformal reserve\n"
              "           will clip to control.min_ceiling and the battery will age almost\n"
              "           purely by calendar. In this regime no scheduling policy can beat\n"
              "           charging to that floor, and a kill test measures the floor, not the\n"
              "           schedule. Size the battery in days of load, not in Wh.")
    summary["energy_per_action_j"] = dict(
        sleep_per_slot=energy.sleep_j, capture_day=energy.capture_day_j,
        capture_night=energy.capture_night_j, triage=energy.triage_j, store=energy.store_j,
        tier_b_wake=energy.wake_j, **{f"refine_{k}": v for k, v in energy.refine_j.items()})
    summary["solar_source"] = cfg.solar.source
    write_json(f"{cfg.results_dir}/tables/environment_summary.json", summary)
    print(f"\n  energy per action (J): {summary['energy_per_action_j']}")


if __name__ == "__main__":
    main()
