#!/usr/bin/env python
"""Figures and outputs/results/REPORT.md from whatever tables exist.

    python pipeline/07_make_report.py

Safe to re-run at any time; it only reads files. Sections whose inputs are
missing are skipped with a note, so a partial run still produces a report.
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sunsched.cli import base_parser, load_cfg, read_csv

COLORS = {"always_now": "#9b333e", "triage_only": "#b8a07a", "ee_now": "#3b4453",
          "ceiling_now": "#6c7688", "lazy_defer": "#116460", "sunsched": "#c77d0a",
          "sunsched_no_defer": "#e0b060", "sunsched_no_ceiling": "#d89a3a",
          "sunsched_no_conformal": "#a86a10", "sunsched_lite": "#f0c880"}
MAIN = ["always_now", "triage_only", "ee_now", "ceiling_now", "lazy_defer", "sunsched"]


def load(path):
    return read_csv(path) if os.path.exists(path) else None


def fmt(v, lo=None, hi=None, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    s = f"{v:.{d}f}"
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{lo:.{d}f}, {hi:.{d}f}]"
    return s


def fig_diel(rows, path):
    h = [int(r["hour"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(h, [r["event_share"] for r in rows], color="#3b4453", label="animal captures")
    ax.plot(h, [r["energy_share"] for r in rows], color="#c77d0a", lw=2.5, label="solar energy")
    ax.set_xlabel("local clock hour")
    ax.set_ylabel("share of daily total")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False)
    ax.set_title("When work arrives vs when energy arrives (CCT20, all cameras)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_frontier(rows, ratio, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    groups = sorted({(r["policy"], r.get("tag_sweep", "")) for r in rows})
    styles = ["o-", "s--", "^:", "d-."]
    for i, (pol, sweep) in enumerate(groups):
        rr = sorted((r for r in rows if r["policy"] == pol and r.get("tag_sweep", "") == sweep
                     and r["ratio"] == ratio), key=lambda r: r["battery_years_to_eol"])
        if not rr:
            continue
        x = [r["battery_years_to_eol"] for r in rr]
        y = [r["value_score"] for r in rr]
        style = styles[sum(1 for p, _ in groups[:i] if p == pol) % len(styles)]
        ax.plot(x, y, style, color=COLORS.get(pol, "k"), label=f"{pol} ({sweep})", ms=4, lw=1.5)
    ax.set_xlabel("battery years to end of life")
    ax.set_ylabel("value score")
    ax.set_title(f"Accuracy vs battery life, harvest/demand = {ratio}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_regime(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for pol in [p for p in MAIN if any(r["policy"] == p for r in rows)]:
        rr = sorted((r for r in rows if r["policy"] == pol), key=lambda r: r["ratio"])
        x = [r["ratio"] for r in rr]
        for ax, m in zip(axes, ["value_score", "battery_years_to_eol"]):
            ax.plot(x, [r[m] for r in rr], "o-", color=COLORS.get(pol, "k"), label=pol, ms=4)
            ax.fill_between(x, [r[f"{m}_lo"] for r in rr], [r[f"{m}_hi"] for r in rr],
                            color=COLORS.get(pol, "k"), alpha=0.12, lw=0)
    for ax, lab in zip(axes, ["value score", "battery years to end of life"]):
        ax.set_xscale("log")
        ax.set_xlabel("harvest-to-demand ratio")
        ax.set_ylabel(lab)
        ax.axvline(1.0, color="k", lw=0.8, ls=":")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_routes(summary, path):
    ratios = sorted({r["ratio"] for r in summary})
    fig, axes = plt.subplots(1, len(ratios), figsize=(4.2 * len(ratios), 3.8), squeeze=False)
    parts = [("triage_final_frac", "triage label", "#b8a07a"),
             ("refined_now_frac", "refined now", "#3b4453"),
             ("refined_later_frac", "refined later", "#c77d0a"),
             ("dropped_frac", "missed", "#9b333e")]
    for ax, ratio in zip(axes[0], ratios):
        rr = [r for r in summary if r["ratio"] == ratio and r["policy"] in MAIN]
        rr.sort(key=lambda r: MAIN.index(r["policy"]))
        bottom = np.zeros(len(rr))
        for key, lab, col in parts:
            vals = np.array([r.get(key, 0.0) if np.isfinite(r.get(key, np.nan)) else 0.0 for r in rr])
            ax.bar(range(len(rr)), vals, bottom=bottom, color=col, label=lab)
            bottom += vals
        ax.set_xticks(range(len(rr)))
        ax.set_xticklabels([r["policy"] for r in rr], rotation=40, ha="right", fontsize=8)
        ax.set_title(f"harvest/demand = {ratio}")
    axes[0][0].set_ylabel("share of captures")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg = load_cfg(args)
    T, F = f"{cfg.results_dir}/tables", f"{cfg.results_dir}/figures"
    L = []
    A = L.append

    env = json.load(open(f"{T}/environment_summary.json")) if os.path.exists(f"{T}/environment_summary.json") else {}
    A("# SunSched results")
    A("")
    A(f"Generated {datetime.now():%Y-%m-%d %H:%M}. Irradiance source: **{env.get('solar_source', '?')}**. "
      f"Preset: **{'quick' if args.quick else 'full'}**. Site: {cfg.experiment.main_site}. "
      f"Weather years: {list(cfg.solar.years)}.")
    if env.get("solar_source") == "analytic":
        A("")
        A("> **Synthetic irradiance.** These results use the offline analytic generator, not measured "
          "weather. Do not report them as real-irradiance results.")
    if args.quick:
        A("")
        A("> **Quick preset.** Subset of images, 2 weather years, a 90-day season. For checking the "
          "pipeline, not for reporting.")

    diel = load(f"{T}/diel_mismatch.csv")
    A("\n## 1. Why schedule in time\n")
    if diel and env:
        fig_diel(diel, f"{F}/diel_mismatch.png")
        A(f"Across {env['n_animal_captures']:,} animal captures from all 20 CCT20 cameras, "
          f"**{env['events_sun_below_horizon'] * 100:.1f}%** arrive with the sun below the horizon, "
          f"when harvest is zero. Only **{env['events_sun_above_40deg'] * 100:.1f}%** arrive with the sun "
          f"above 40 degrees, which delivers **{env['energy_sun_above_40deg'] * 100:.1f}%** of the energy. "
          f"Hourly correlation between the two: {env['pearson_hourly_events_vs_energy']:.2f}.")
        A("\n![diel mismatch](figures/diel_mismatch.png)")
        A(f"\nEnergy per action (J): `{env.get('energy_per_action_j')}`")
    else:
        A("_Run pipeline/03_build_environment.py._")

    grid = load(f"{T}/accuracy_grid.csv")
    A("\n## 2. Classifier (frozen MobileNetV3-Large, trained exit heads)\n")
    if grid:
        A("Evaluated on the 9 held-out cameras (trans_test); temperatures fitted on trans_val.\n")
        A("| input height | exit | MMACs | accuracy | macro-F1 | animal-vs-empty | ECE before -> after |")
        A("|---|---|---|---|---|---|---|")
        for r in grid:
            A(f"| {int(r['resolution'])} | {int(r['exit'])} | {r['mmacs']:.1f} | {r['acc_eval']:.3f} | "
              f"{r['macro_f1_eval']:.3f} | {r['animal_vs_empty_acc_eval']:.3f} | "
              f"{r['ece_calib_before']:.3f} -> {r['ece_calib_after']:.3f} |")
    else:
        A("_Run pipeline/02_train_exits.py._")

    summary = load(f"{T}/summary.csv")
    A("\n## 3. Main comparison\n")
    if summary:
        fig_routes(summary, f"{F}/routes.png")
        A("Mean over held-out cameras x weather years, with 95% bootstrap CIs.")
        for ratio in sorted({r["ratio"] for r in summary}):
            A(f"\n**Harvest-to-demand ratio {ratio}**\n")
            A("| policy | value score | animal recall | missed | latency p95 (h) | battery years | time SoC>90% | wakes/day |")
            A("|---|---|---|---|---|---|---|---|")
            for r in sorted((r for r in summary if r["ratio"] == ratio), key=lambda r: -r["value_score"]):
                A(f"| {r['policy']} | {fmt(r['value_score'], r['value_score_lo'], r['value_score_hi'])} | "
                  f"{fmt(r['animal_recall'])} | {fmt(r['dropped_frac'])} | {fmt(r['latency_p95_h'], d=1)} | "
                  f"{fmt(r['battery_years_to_eol'], r['battery_years_to_eol_lo'], r['battery_years_to_eol_hi'], d=2)} | "
                  f"{fmt(r['frac_time_soc_above_90'] * 100, d=1)}% | {fmt(r['b_wakes_per_day'], d=2)} |")
        A("\n![routes](figures/routes.png)")
    else:
        A("_Run pipeline/04_run_experiments.py._")

    kill = f"{T}/kill_test.json"
    A("\n## 4. Kill test\n")
    if os.path.exists(kill):
        k = json.load(open(kill))
        A(f"**{'PASS' if k['passed'] else 'FAIL'}** against {', '.join(k['baselines'])} "
          f"(margins: {k['margins']}). Per regime: {k['per_ratio']}.")
        kt = load(f"{T}/kill_test.csv")
        if kt:
            A("\n| ratio | versus | value diff | p (Holm) | life ratio | p (Holm) | beats |")
            A("|---|---|---|---|---|---|---|")
            for r in kt:
                A(f"| {r['ratio']} | {r['versus']} | {r['value_diff']:+.3f} | {r['p_value_holm']:.3g} | "
                  f"x{r['life_ratio']:.2f} | {r['p_life_holm']:.3g} | {r['beats']} |")
    else:
        A("_Run pipeline/06_kill_test.py._")

    fr = load(f"{T}/frontier.csv")
    A("\n## 5. Accuracy vs battery-life frontiers\n")
    if fr:
        for ratio in sorted({r["ratio"] for r in fr}):
            p = f"frontier_{ratio}.png"
            fig_frontier(fr, ratio, f"{F}/{p}")
            A(f"![frontier {ratio}](figures/{p})")
    else:
        A("_Run pipeline/05_sweeps.py --only frontier._")

    rg = load(f"{T}/regime.csv")
    A("\n## 6. Regime: where scheduling matters\n")
    if rg:
        fig_regime(rg, f"{F}/regime.png")
        A("![regime](figures/regime.png)")
    else:
        A("_Run pipeline/05_sweeps.py --only regime._")

    sens = load(f"{T}/sensitivity.csv")
    A("\n## 7. Sensitivity to assumed constants\n")
    if sens:
        A("SunSched minus the best of the strong baselines at each setting "
          "(value score; battery-life ratio).\n")
        A("| assumption | setting | value diff | life ratio |")
        A("|---|---|---|---|")
        for (param, level) in sorted({(r["tag_param"], r["tag_level"]) for r in sens}):
            rr = [r for r in sens if r["tag_param"] == param and r["tag_level"] == level]
            ours = next((r for r in rr if r["policy"] == "sunsched"), None)
            base = [r for r in rr if r["policy"] in ("ee_now", "ceiling_now", "lazy_defer")]
            if not ours or not base:
                continue
            best = max(base, key=lambda r: r["value_score"])
            A(f"| {param} | {level} | {ours['value_score'] - best['value_score']:+.3f} | "
              f"x{ours['battery_years_to_eol'] / max(best['battery_years_to_eol'], 1e-9):.2f} |")
    else:
        A("_Run pipeline/05_sweeps.py --only sensitivity._")

    auto = load(f"{T}/autonomy.csv")
    A("\n## 7b. Battery autonomy: can scheduling matter at all?\n")
    if auto:
        A("Days of load the battery holds, swept as a first-class axis rather than fixed. "
          "`ceil@floor` is the fraction of slots where the conformal ceiling was clipped to "
          "`control.min_ceiling`: near 1, the risk-controlled reserve cannot steer the battery "
          "whatever its coverage, the cell ages by calendar, and the comparison measures that "
          "floor rather than the schedule.\n")
        A("This sweep is **exploratory, not pre-registered**. The kill test in section 4 stands "
          "as the pre-registered result at the default sizing. Every level is reported here "
          "precisely so that no single one is selected after the fact.\n")
        A("| days | ratio | sunsched value | vs ee_now | vs ceiling_now | vs lazy_defer | life x best | ceil@floor |")
        A("|---|---|---|---|---|---|---|---|")
        for d in sorted({r["tag_value"] for r in auto}):
            for ratio in sorted({r["ratio"] for r in auto}):
                rr = [r for r in auto if r["tag_value"] == d and r["ratio"] == ratio]
                ours = next((r for r in rr if r["policy"] == "sunsched"), None)
                base = {r["policy"]: r for r in rr
                        if r["policy"] in ("ee_now", "ceiling_now", "lazy_defer")}
                if not ours or len(base) < 3:
                    continue
                best = max(base.values(), key=lambda r: r["value_score"])
                dv = lambda p: ours["value_score"] - base[p]["value_score"]
                A(f"| {d:g} | {ratio:g} | {ours['value_score']:.3f} | {dv('ee_now'):+.3f} "
                  f"| {dv('ceiling_now'):+.3f} | {dv('lazy_defer'):+.3f} "
                  f"| x{ours['battery_years_to_eol'] / max(best['battery_years_to_eol'], 1e-9):.2f} "
                  f"| {ours['ceiling_at_floor_frac']:.2f} |")
    else:
        A("_Run pipeline/05_sweeps.py --only autonomy._")

    A("\n## 8. What is assumed, not measured\n")
    A("- No hardware: node power figures are datasheet-typical assumptions (`NodeCfg`), swept in section 7.")
    A("- Battery ageing uses the Xu et al. (IEEE Trans. Smart Grid 2018) semi-empirical model; "
      "constants must be checked against the paper and are swept +/-50%.")
    A("- Enclosure heating (battery temperature above air temperature) is an assumption, swept 0-30 C.")
    A("- Camera coordinates are region-level (CCT does not publish them); capture clock times may "
      "include daylight saving time.")
    A("- Captures from several calendar years at one camera are laid onto one simulated year.")
    A("- MAC counts are measured from the real network; energy per MAC is assumed.")
    A("- `sunsched` and `sunsched_no_conformal` differ only in the reserve forecaster, so compare "
      "them on `reserve_coverage` in `tables/runs.csv`: the conformal bound should sit near "
      "1 - alpha and the point forecast well below it. Meeting that target is not the same as "
      "steering the battery. If `ceiling_at_floor_frac` is near 1, the bound was clipped to "
      "`control.min_ceiling` before it reached the cell, and no lifetime difference may be "
      "credited to the reserve however good its coverage looks. That is what happens when the "
      "battery holds many days of load: see `days_of_autonomy` in `environment_summary.json`, "
      "and the `control.min_ceiling` and `battery.capacity_wh` rows of section 7.")

    with open(f"{cfg.results_dir}/REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {cfg.results_dir}/REPORT.md and figures in {F}")


if __name__ == "__main__":
    main()
