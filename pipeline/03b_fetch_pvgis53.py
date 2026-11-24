#!/usr/bin/env python
"""WP2: PVGIS 5.3 hourly weather, 2019-2023, for the six SolCascade climates."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sunsched.cli import banner, base_parser, write_csv
from sunsched.config import SITES
from sunsched.env.solar import fetch_pvgis, pvgis_year

CLIMATES = {"C1": "annamites", "C2": "meta", "C3": "white_mtns", "C4": "skukuza", "C4b": "nkhotakota", "C5": "wellington", "C6": "bergen"}
YEARS = (2019, 2020, 2021, 2022, 2023)
V53 = ("https://re.jrc.ec.europa.eu/api/v5_3/seriescalc",)


def check(site, data):
    """Rows of summary numbers; raises if a year is unusable."""
    s, rows = SITES[site], []
    hourly = data["outputs"]["hourly"]
    db = data.get("inputs", {}).get("meteo_data", {}).get("radiation_db", "?")
    expected = sum(8784 if y % 4 == 0 else 8760 for y in YEARS)
    if len(hourly) != expected:
        raise SystemExit(f"{site}: {len(hourly)} hourly rows, expected {expected}")
    bad = sum(1 for h in hourly if not np.isfinite(float(h.get("G(i)", np.nan))) or not np.isfinite(float(h.get("T2m", np.nan))))
    if bad:
        raise SystemExit(f"{site}: {bad} rows without irradiance or temperature")
    for y in YEARS:
        sy = pvgis_year(site, s, y, data, 5)
        g, e, t = sy.irradiance, sy.elevation_deg, sy.air_temp_c
        dark = float(g[e < -6].sum() / max(g.sum(), 1e-9))
        noon = float(np.sum(g * ((np.arange(len(g)) % 288) / 12.0)) / max(g.sum(), 1e-9))
        kwh = float(g.sum() * 300 / 3.6e6)
        if dark > 0.01 or not 400 < kwh < 3500 or not 10.5 < noon < 13.8:
            raise SystemExit(f"{site} {y}: implausible (dark share {dark:.3f}, {kwh:.0f} kWh/m2, centre {noon:.2f} h)")
        rows.append(dict(site=site, lat=s["lat"], lon=s["lon"], year=y, radiation_db=db, kwh_per_m2=round(kwh, 1),
                         worst_30d_share=round(float(np.convolve(g, np.ones(30 * 288), "valid").min() / g.sum()), 4),
                         temp_mean=round(float(t.mean()), 2), temp_min=round(float(t.min()), 1),
                         temp_max=round(float(t.max()), 1), hours_below_0c=round(float((t < 0).sum() / 12), 0),
                         energy_centre_h=round(noon, 2), dark_share=round(dark, 5)))
    return rows


def main():
    args = base_parser(__doc__).parse_args()
    cache = f"{args.data}/pvgis53"
    banner("PVGIS 5.3, 2019-2023")
    rows = []
    for c, site in CLIMATES.items():
        s = SITES[site]
        time.sleep(3)
        data = fetch_pvgis(s["lat"], s["lon"], YEARS[0], YEARS[-1], cache, endpoints=V53)
        r = check(site, data)
        rows += [dict(climate=c, **x) for x in r]
        print(f"  {c} {site:11} {r[0]['radiation_db']:14} " + "  ".join(f"{x['year']}:{x['kwh_per_m2']:.0f}" for x in r),
              f" min {min(x['temp_min'] for x in r)} C", flush=True)
    write_csv(f"{args.out}/weather/pvgis53_summary.csv", rows)


if __name__ == "__main__":
    main()
