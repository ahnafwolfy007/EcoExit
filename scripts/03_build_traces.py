#!/usr/bin/env python
"""Build solar traces, event streams, and conformal forecast calibration for
every site (report section 09 / 11: train on Dhaka, test on 4 held-out
climates). Fully offline (analytic clear-sky + AR(1) cloud process) so the
whole repo runs with no network beyond the one-time CIFAR-100 download.

Usage:
    python scripts/03_build_traces.py

Produces:
    artifacts/traces/<site>.npz         -- ghi, ghi_clear, elevation, harvest_j, kc, stream
    artifacts/traces/<site>_forecast.npz -- point/actual/lower-bound harvest per block, coverage
    figures/solar_traces.png
    figures/forecast_coverage_<train_site>.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecoexit.config import Config, SITES, TRAIN_SITE, TEST_SITES
from ecoexit.solar.traces import make_trace
from ecoexit.sim.stream import make_stream
from ecoexit.forecast.conformal import simulate_forecast_and_calibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--test-set-size", type=int, default=10000)
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(f"{args.artifacts}/traces", exist_ok=True)
    os.makedirs(f"{args.out}/figures", exist_ok=True)

    cfg = Config()
    if args.days:
        cfg.solar.days = args.days
    slots_per_day = int(86400 / cfg.solar.slot_seconds)
    horizon_slots = int(cfg.control.horizon_hours * 3600 / cfg.solar.slot_seconds)

    all_sites = [TRAIN_SITE] + TEST_SITES
    traces = {}

    fig, axes = plt.subplots(len(all_sites), 1, figsize=(9, 2.1 * len(all_sites)), sharex=True)
    for i, site in enumerate(all_sites):
        t = make_trace(site, SITES[site], cfg.solar, seed=cfg.solar.seed + i)
        stream = make_stream(t.elevation_deg, t.n_slots, cfg.solar.slot_seconds,
                              cfg.stream, dataset_size=args.test_set_size,
                              seed=cfg.stream.seed + i)
        traces[site] = (t, stream)

        np.savez(f"{args.artifacts}/traces/{site}.npz",
                  ghi=t.ghi, ghi_clear=t.ghi_clear, elevation=t.elevation_deg,
                  harvest_j=t.harvest_j, kc=t.kc,
                  stream_value=stream.value, stream_is_event=stream.is_event,
                  stream_cifar_index=stream.cifar_index)

        days_axis = np.arange(t.n_slots) * cfg.solar.slot_seconds / 86400.0
        axes[i].fill_between(days_axis, 0, t.ghi, color="#9A6508", alpha=0.55, lw=0)
        axes[i].plot(days_axis, t.ghi_clear, color="#3B4453", lw=0.8, alpha=0.7)
        ev_days = days_axis[stream.is_event]
        axes[i].scatter(ev_days, np.full(ev_days.shape, t.ghi_clear.max() * 1.05),
                         marker="|", color="#9B333E", s=40, label="event" if i == 0 else None)
        axes[i].set_ylabel(f"{site}\nW/m²", fontsize=9)
        kc_mean = t.kc[t.ghi_clear > 1].mean()
        axes[i].text(0.99, 0.85, f"mean kc={kc_mean:.2f}", transform=axes[i].transAxes,
                     ha="right", fontsize=8, color="#3B4453")
        print(f"[traces] {site:8s} n_slots={t.n_slots} days={cfg.solar.days} "
              f"mean_kc={kc_mean:.3f} n_events={stream.is_event.sum()}")
    axes[-1].set_xlabel("day")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Synthesised solar irradiance (shaded) vs clear-sky envelope (line), by site")
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/solar_traces.png", dpi=150)
    plt.close(fig)
    print(f"[traces] wrote {args.out}/figures/solar_traces.png")

    # -- conformal forecast calibration, fit on the TRAIN site only --------
    t_train, _ = traces[TRAIN_SITE]
    out = simulate_forecast_and_calibration(
        t_train.kc, slots_per_day, horizon_slots, t_train.ghi_clear,
        cfg.solar.panel_area_m2, cfg.solar.panel_efficiency, cfg.solar.slot_seconds,
        cfg.control.alpha, cfg.control.aci_gamma,
    )
    cov = out["aci"].empirical_coverage()
    print(f"[traces] {TRAIN_SITE} (train): target coverage={1-cfg.control.alpha:.2f}, "
          f"empirical={cov:.3f} over {len(out['point_j'])} blocks")

    np.savez(f"{args.artifacts}/traces/{TRAIN_SITE}_forecast.npz",
              point_j=out["point_j"], actual_j=out["actual_j"], lower_j=out["lower_j"],
              alpha_t=out["alpha_t"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
    xb = np.arange(len(out["point_j"]))
    ax1.plot(xb, out["actual_j"] / 1000, "o-", color="#151A23", ms=4, label="actual harvest")
    ax1.plot(xb, out["point_j"] / 1000, "s--", color="#9A6508", ms=4, label="point forecast")
    ax1.fill_between(xb, out["lower_j"] / 1000, out["point_j"] / 1000,
                       color="#9A6508", alpha=0.15, label="conformal margin")
    ax1.plot(xb, out["lower_j"] / 1000, ":", color="#9B333E", lw=1.5, label="calibrated lower bound")
    ax1.set_ylabel("kJ per block")
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_title(f"Conformal harvest bound, {TRAIN_SITE} "
                   f"(target coverage {1-cfg.control.alpha:.0%}, achieved {cov:.0%})")

    running_cov = np.cumsum(out["actual_j"] >= out["lower_j"]) / (xb + 1)
    ax2.plot(xb, running_cov, color="#116460")
    ax2.axhline(1 - cfg.control.alpha, color="#9B333E", ls="--", lw=1, label="target")
    ax2.set_ylabel("running coverage")
    ax2.set_xlabel("planning block")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/figures/forecast_coverage_{TRAIN_SITE}.png", dpi=150)
    plt.close(fig)
    print(f"[traces] wrote {args.out}/figures/forecast_coverage_{TRAIN_SITE}.png")

    # -- also compute + save lower-bound harvest series for every site, using
    #    an ACI state *warmed up on the training site* then continued at test
    #    time (this is the honest cross-climate protocol: the forecaster is
    #    not re-fit per site, only its running alpha_t continues to adapt).
    for site in all_sites:
        t, _ = traces[site]
        out_s = simulate_forecast_and_calibration(
            t.kc, slots_per_day, horizon_slots, t.ghi_clear,
            cfg.solar.panel_area_m2, cfg.solar.panel_efficiency, cfg.solar.slot_seconds,
            cfg.control.alpha, cfg.control.aci_gamma,
        )
        cov_s = out_s["aci"].empirical_coverage()
        np.savez(f"{args.artifacts}/traces/{site}_forecast.npz",
                  point_j=out_s["point_j"], actual_j=out_s["actual_j"],
                  lower_j=out_s["lower_j"], alpha_t=out_s["alpha_t"])
        print(f"[traces] {site:8s} forecast coverage={cov_s:.3f} "
              f"(n_blocks={len(out_s['point_j'])})")

    print("[traces] done.")


if __name__ == "__main__":
    main()
