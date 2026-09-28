#!/usr/bin/env python
"""Figures and outputs/<corpus>/results/REPORT.md from whatever tables exist.

    python pipeline/07_make_report.py --corpus serengeti

Safe to re-run at any time; it only reads files. Sections whose inputs are
missing are skipped with a note.
"""
import json
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sunsched.cli import base_parser, load_cfg, read_csv

MAIN = ["always_now", "triage_only", "ee_now", "ceiling_now", "lazy_defer", "lazy_animal", "sunsched"]
H5_BASE = ["always_now", "triage_only", "ee_now", "ceiling_now", "lazy_defer", "lazy_animal"]


def load(path):
    return read_csv(path) if os.path.exists(path) else None


def loadj(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def cell(summary, a, r, p):
    for s in summary:
        if s["policy"] == p and math.isclose(s["autonomy"], a) and math.isclose(s["ratio"], r):
            return s
    return None


def fmt(v, d=3):
    return "-" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def heatmap(ax, grid, autonomies, ratios, title, fmtstr, cmap, center=None):
    data = np.array(grid, dtype=float)
    if center is None:
        im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower")
    else:
        span = np.nanmax(np.abs(data - center)) if np.isfinite(data).any() else 1.0
        im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower",
                       vmin=center - span, vmax=center + span)
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"{r:g}" for r in ratios])
    ax.set_yticks(range(len(autonomies)))
    ax.set_yticklabels([f"{a:g}" for a in autonomies])
    ax.set_xlabel("harvest-to-demand ratio")
    ax.set_ylabel("battery autonomy (days)")
    ax.set_title(title, fontsize=10)
    for i in range(len(autonomies)):
        for j in range(len(ratios)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, fmtstr.format(data[i, j]), ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)


def fig_regime(summary, autonomies, ratios, path):
    def g(f):
        return [[f(a, r) for r in ratios] for a in autonomies]

    def diff(a, r, p, q, m):
        x, y = cell(summary, a, r, p), cell(summary, a, r, q)
        return x[m] - y[m] if x and y else np.nan

    def ratio(a, r, p, q, m):
        x, y = cell(summary, a, r, p), cell(summary, a, r, q)
        return x[m] / max(y[m], 1e-9) if x and y else np.nan

    def vs_best(a, r):
        s = cell(summary, a, r, "sunsched")
        b = [cell(summary, a, r, p) for p in H5_BASE]
        b = [x["value_score"] for x in b if x]
        return s["value_score"] - max(b) if s and b else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    heatmap(axes[0], g(lambda a, r: diff(a, r, "lazy_defer", "always_now", "value_score")),
            autonomies, ratios, "H2: value, lazy_defer - always_now", "{:+.3f}", "RdBu", 0.0)
    heatmap(axes[1], g(lambda a, r: ratio(a, r, "ceiling_now", "always_now", "battery_years_to_eol")),
            autonomies, ratios, "H3: battery life, ceiling_now / always_now", "x{:.2f}", "RdBu", 1.0)
    heatmap(axes[2], g(vs_best), autonomies, ratios,
            "H5: value, sunsched - best baseline", "{:+.3f}", "RdBu", 0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_diel(rows, path, corpus):
    h = [int(r["hour"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(h, [r["event_share"] for r in rows], color="#3b4453", label="animal captures")
    ax.plot(h, [r["energy_share"] for r in rows], color="#c77d0a", lw=2.5, label="solar energy")
    ax.set_xlabel("local clock hour")
    ax.set_ylabel("share of daily total")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False)
    ax.set_title(f"When work arrives vs when energy arrives ({corpus})")
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

    env = loadj(f"{T}/environment_summary.json") or {}
    man = loadj(f"{T}/run_manifest.json") or {}
    hyp = loadj(f"{T}/hypotheses.json")
    decides = cfg.data.corpus == "serengeti" and not args.quick
    A(f"# SunSched results: {cfg.data.corpus}")
    A("")
    A(f"Generated {datetime.now():%Y-%m-%d %H:%M}. Corpus role: "
      f"**{'TEST (pre-registered; this run decides)' if decides else 'development (informative only)'}**. "
      f"Weather: **{env.get('solar_source', '?')}**. Preset: **{'quick' if args.quick else 'full'}**. "
      f"Code: `{man.get('git', {}).get('commit', '?')[:10]}`"
      f"{' (UNCOMMITTED CHANGES)' if man.get('git', {}).get('dirty') else ''}.")
    if env.get("solar_source") == "analytic":
        A("\n> **Synthetic weather.** Not a real-irradiance result.")
    if args.quick:
        A("\n> **Quick preset.** Pipeline check only; the numbers below are not results.")

    A("\n## 1. Hypotheses (PREREGISTRATION.md)\n")
    if hyp:
        A(f"| H1 mismatch | H2 deferral, small storage | H3 ceiling, large storage | H4 SunSched, small | H5 SunSched near-best |")
        A("|---|---|---|---|---|")
        A("| " + " | ".join("holds" if hyp[k] else "fails" for k in ("H1", "H2", "H3", "H4", "H5")) + " |")
        A(f"\n**Primary claim (H1-H3): {'HOLDS' if hyp['primary'] else 'DOES NOT HOLD'}.** {hyp['claim']}")
        if not decides:
            A("\nThis corpus does not decide the hypotheses; only the full Serengeti run does.")
    else:
        A("_Run pipeline/06_hypotheses.py._")

    diel = load(f"{T}/diel_mismatch.csv")
    A("\n## 2. When animals arrive vs when energy arrives (H1)\n")
    if diel and env:
        fig_diel(diel, f"{F}/diel_mismatch.png", cfg.data.corpus)
        ci = env.get("events_sun_below_horizon_ci", [np.nan, np.nan])
        A(f"Across {env['n_animal_captures']:,} animal captures from {env.get('n_cameras', '?')} cameras, "
          f"**{env['events_sun_below_horizon'] * 100:.1f}%** (95% CI over cameras {ci[0] * 100:.1f}-{ci[1] * 100:.1f}%) "
          f"arrive with the sun below the horizon. **{env['events_sun_above_40deg'] * 100:.1f}%** arrive with the "
          f"sun above 40 degrees, which delivers **{env['energy_sun_above_40deg'] * 100:.1f}%** of the energy. "
          f"Hourly correlation: {env['pearson_hourly_events_vs_energy']:.2f}.")
        A("\n![diel](figures/diel_mismatch.png)")
    else:
        A("_Run pipeline/03_build_environment.py._")

    grid = load(f"{T}/accuracy_grid.csv")
    A("\n## 3. Classifier and triage detector\n")
    if grid:
        A("| input height | exit | MMACs | accuracy | macro-F1 | animal vs empty | top-1 conf p10-p90 |")
        A("|---|---|---|---|---|---|---|")
        for r in grid:
            ex = "detector" if r.get("tap") == "detector" else f"{int(r['exit'])}"
            A(f"| {fmt(r.get('resolution'), 0)} | {ex} | "
              f"{fmt(r.get('mmacs'), 1)} | {fmt(r.get('acc_eval'))} | {fmt(r.get('macro_f1_eval'))} | "
              f"{fmt(r.get('animal_vs_empty_acc_eval'))} | {fmt(r.get('top1_conf_p10'))}-{fmt(r.get('top1_conf_p90'))} |")
    summary = load(f"{T}/summary.csv")
    A("\n## 4. Regime map\n")
    if summary:
        auts = sorted({s["autonomy"] for s in summary if np.isfinite(s["autonomy"])})
        rats = sorted({s["ratio"] for s in summary})
        fig_regime(summary, auts, rats, f"{F}/regime_map.png")
        A("Left: where deferral beats classifying immediately (blue = deferral better). Middle: how much "
          "longer the battery lasts with a charge ceiling. Right: SunSched v2 against the best baseline "
          "in each cell.\n")
        A("![regime map](figures/regime_map.png)")
        A("\n### Every policy at four representative cells\n")
        picks = list(dict.fromkeys([(auts[0], 1.0), (auts[min(2, len(auts) - 1)], 1.5),
                                    (auts[-2] if len(auts) > 1 else auts[-1], 1.5), (auts[-1], 2.0)]))
        for a, r in picks:
            rows = [cell(summary, a, r, p) for p in MAIN + ["sunsched_no_gate", "sunsched_no_defer",
                                                          "sunsched_no_ceiling", "sunsched_no_conformal"]]
            rows = [x for x in rows if x]
            if not rows:
                continue
            A(f"\n**Autonomy {a:g} days, ratio {r:g}**\n")
            A("| policy | value | animal recall | missed | refined now | refined later | latency p95 h | battery years | SoC>90% |")
            A("|---|---|---|---|---|---|---|---|---|")
            for x in sorted(rows, key=lambda x: -x["value_score"]):
                A(f"| {x['policy']} | {fmt(x['value_score'])} [{fmt(x['value_score_lo'])}, {fmt(x['value_score_hi'])}] | "
                  f"{fmt(x['animal_recall'])} | {fmt(x['dropped_frac'])} | {fmt(x['refined_now_frac'])} | "
                  f"{fmt(x['refined_later_frac'])} | {fmt(x['latency_p95_h'], 1)} | {fmt(x['battery_years_to_eol'], 1)} | "
                  f"{fmt(x['frac_time_soc_above_90'] * 100, 1)}% |")
    else:
        A("_Run pipeline/04_run_experiments.py._")

    sens = load(f"{T}/sensitivity.csv")
    A("\n## 5. Do the regime effects survive every assumption?\n")
    if sens:
        A("At each probe cell and each assumption extreme: value of lazy_defer minus always_now (the H2 "
          "effect) and battery-life ratio of ceiling_now to always_now (the H3 effect).\n")
        A("| assumption | setting | autonomy | ratio | H2 effect | H3 effect |")
        A("|---|---|---|---|---|---|")
        keys = sorted({(r["tag_param"], r["tag_level"], r["autonomy"], r["ratio"]) for r in sens})
        for p, lvl, a, r in keys:
            get = lambda pol: next((x for x in sens if x["tag_param"] == p and x["tag_level"] == lvl
                                    and x["autonomy"] == a and x["ratio"] == r and x["policy"] == pol), None)
            la, al, ce = get("lazy_defer"), get("always_now"), get("ceiling_now")
            if not (la and al and ce):
                continue
            A(f"| {p} | {lvl} | {a:g} | {r:g} | {la['value_score'] - al['value_score']:+.3f} | "
              f"x{ce['battery_years_to_eol'] / max(al['battery_years_to_eol'], 1e-9):.2f} |")
    else:
        A("_Run pipeline/05_sweeps.py._")

    A("\n## 6. Assumed, not measured\n")
    A("- No hardware: node power figures (`NodeCfg`) are datasheet-typical assumptions, swept in section 5.")
    A("- Battery ageing uses Xu et al. (IEEE Trans. Smart Grid 2018); its constants must be checked "
      "against the paper and are swept +/-50%.")
    A("- Enclosure heating above air temperature is an assumption, swept 0-30 C.")
    A("- Camera coordinates are region-level; capture clock times may include daylight saving time.")
    A("- Captures from several calendar years at one camera are laid onto one simulated year.")
    A("- The frozen ImageNet trunk is a weak species classifier on unseen cameras; absolute value "
      "scores are low. The hypotheses concern differences between policies sharing that classifier.")

    with open(f"{cfg.results_dir}/REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {cfg.results_dir}/REPORT.md and figures in {F}")


if __name__ == "__main__":
    main()
