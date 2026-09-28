#!/usr/bin/env python
"""Weather per site-year, capture streams per camera, and the H1 measurement.

    python pipeline/03_build_environment.py --corpus cct20
    python pipeline/03_build_environment.py --corpus cct20 --solar analytic   # offline fallback, labelled synthetic

Writes outputs/<corpus>/artifacts/solar/<site>_<year>.npz and events.npz, and
tables diel_mismatch.csv, sizing.csv and environment_summary.json. The last of
these carries the H1 statistic: the share of animal captures that arrive with
the sun below the horizon, over every camera of the corpus.
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sunsched.cli import banner, base_parser, load_cfg, write_csv, write_json
from sunsched.config import SITES, pvgis_range
from sunsched.data.corpus import load_corpus
from sunsched.env import solar as S
from sunsched.env.events import build_streams, diel_profile, slot_in_year
from sunsched.experiment import always_now_demand_j, essential_j_per_day
from sunsched.sim.node import build_node_energy
from sunsched.sim.outcomes import load_outcomes


def build_solar(cfg) -> dict:
    out_dir = f"{cfg.artifacts_dir}/solar"
    os.makedirs(out_dir, exist_ok=True)
    start, end = pvgis_range()
    years = {}
    for site, s in SITES.items():
        data = None
        if cfg.solar.source == "pvgis":
            try:
                data = S.fetch_pvgis(s["lat"], s["lon"], start, end, cfg.data.pvgis_dir)
            except Exception as e:
                raise SystemExit(f"PVGIS unavailable for {site}: {e}\n"
                                 f"  Fix the network and re-run, or use --solar analytic and report\n"
                                 f"  the results as synthetic-weather results.")
        for y in cfg.solar.years:
            sy = (S.pvgis_year(site, s, y, data, cfg.solar.slot_minutes) if data is not None
                  else S.analytic_year(site, s, y, cfg.solar.slot_minutes))
            S.save_year(sy, f"{out_dir}/{site}_{y}.npz")
            years[(site, y)] = sy
        e = np.mean([S.irradiation_j_per_m2(years[(site, y)].irradiance, cfg.solar, cfg.slot_seconds)
                     for y in cfg.solar.years])
        t = np.mean([years[(site, y)].air_temp_c.mean() for y in cfg.solar.years])
        print(f"  {site:<11} {cfg.solar.source}: {e / 3.6e6 / 365:6.3f} kWh/m^2/day electrical, air {t:5.1f} C")
    return years


def diel_mismatch(cfg, corpus, years) -> dict:
    site_name = cfg.experiment.main_site
    site = SITES[site_name]
    recs = [r for r in corpus.all_records if r.label != cfg.task.empty_class]
    doy = np.array([r.timestamp.timetuple().tm_yday for r in recs])
    clock = np.array([r.timestamp.hour + r.timestamp.minute / 60.0 for r in recs])
    elev = S.elevation_at(site["lat"], site["lon"], site["utc_offset"], doy, clock)

    spd = cfg.slots_per_day
    ev_share = diel_profile(clock.astype(int))
    energy = np.zeros(24)
    irr_all, elev_all = [], []
    for y in cfg.solar.years:
        sy = years[(site_name, y)]
        hours = ((np.arange(sy.n_slots) % spd) * cfg.solar.slot_minutes // 60).astype(int)
        energy += np.bincount(hours, weights=sy.irradiance, minlength=24)
        irr_all.append(sy.irradiance)
        elev_all.append(sy.elevation_deg)
    energy_share = energy / energy.sum()
    irr_all, elev_all = np.concatenate(irr_all), np.concatenate(elev_all)
    write_csv(f"{cfg.results_dir}/tables/diel_mismatch.csv",
              [dict(hour=h, event_share=round(float(ev_share[h]), 5),
                    energy_share=round(float(energy_share[h]), 5)) for h in range(24)])

    def energy_in(lo, hi):
        m = (elev_all >= lo) & (elev_all < hi)
        return float(irr_all[m].sum() / irr_all.sum())

    # Bootstrap over cameras, so the interval reflects camera-to-camera variation
    # rather than treating every frame of one busy camera as independent.
    rng = np.random.default_rng(0)
    locs = np.array([r.location for r in recs])
    uniq = np.unique(locs)
    night = elev < 0
    per_loc = {l: night[locs == l] for l in uniq}
    boots = []
    for _ in range(1000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        boots.append(np.concatenate([per_loc[l] for l in pick]).mean())
    out = dict(
        corpus=cfg.data.corpus, site=site_name, n_animal_captures=len(recs), n_cameras=int(len(uniq)),
        events_sun_below_horizon=float(night.mean()),
        events_sun_below_horizon_ci=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        events_sun_below_15deg=float(np.mean(elev < 15)),
        events_sun_above_40deg=float(np.mean(elev >= 40)),
        energy_sun_below_15deg=energy_in(-90, 15),
        energy_sun_above_40deg=energy_in(40, 91),
        pearson_hourly_events_vs_energy=float(np.corrcoef(ev_share, energy_share)[0, 1]),
        note="capture clock times are camera clocks; DST handling is unknown (up to 1 h)",
    )
    print(f"  {len(recs):,} animal captures from {len(uniq)} cameras")
    print(f"  {out['events_sun_below_horizon'] * 100:5.1f}% with the sun below the horizon "
          f"(95% CI over cameras {out['events_sun_below_horizon_ci'][0] * 100:.1f}-"
          f"{out['events_sun_below_horizon_ci'][1] * 100:.1f}%)")
    print(f"  {out['events_sun_above_40deg'] * 100:5.1f}% with the sun above 40 deg, which delivers "
          f"{out['energy_sun_above_40deg'] * 100:.1f}% of the energy; hourly r = "
          f"{out['pearson_hourly_events_vs_energy']:.2f}")
    return out


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

    banner("3. when animals arrive vs when energy arrives (H1)")
    corpus = load_corpus(cfg, quick=args.quick)
    summary = diel_mismatch(cfg, corpus, years)

    banner("4. per-camera energy and sizing")
    energy = build_node_energy(cfg.node, oc.macs, cfg.slot_seconds)
    spd, site = cfg.slots_per_day, cfg.experiment.main_site
    rows = []
    for loc, st in sorted(streams.items(), key=lambda kv: -len(kv[1].frame_idx)):
        n = st.n_days * spd
        window = slice(st.day_offset * spd, st.day_offset * spd + n)
        night = years[(site, cfg.solar.years[0])].elevation_deg[window] < 0
        ess = essential_j_per_day(st.slot, night, n, energy, st.n_days)
        demand = always_now_demand_j(st.slot, night, n, energy)
        row = dict(camera=loc, captures=len(st.frame_idx),
                   captures_per_day=round(len(st.frame_idx) / st.n_days, 2),
                   essential_j_per_day=round(ess, 1),
                   always_now_j_per_day=round(demand / st.n_days, 1),
                   battery_wh_at_1_day=round(ess / 3600, 4),
                   battery_wh_at_30_days=round(30 * ess / 3600, 3))
        rows.append(row)
        print(f"  camera {loc:>6}: {row['captures_per_day']:7.2f} captures/day, essential "
              f"{row['essential_j_per_day']:7.1f} J/day, always-now {row['always_now_j_per_day']:8.1f} J/day")
    write_csv(f"{cfg.results_dir}/tables/sizing.csv", rows)
    summary["energy_per_action_j"] = dict(
        sleep_per_slot=energy.sleep_j, capture_day=energy.capture_day_j,
        capture_night=energy.capture_night_j, triage=energy.triage_j, store=energy.store_j,
        tier_b_wake=energy.wake_j, **{f"refine_{k}": v for k, v in energy.refine_j.items()})
    summary["solar_source"] = cfg.solar.source
    write_json(f"{cfg.results_dir}/tables/environment_summary.json", summary)


if __name__ == "__main__":
    main()
