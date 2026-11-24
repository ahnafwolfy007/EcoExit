"""Tests for SolCascade WP0-WP2 (synthetic data only)."""
import math

import numpy as np
import pandas as pd

from sunsched.config import AgingCfg, Config
from sunsched.data import eligibility as E
from sunsched.env import chemistry as CH
from sunsched.env.battery import aging_summary
from sunsched.policies.base import RunContext
from sunsched.sim.node import build_node_energy
from sunsched.sim.profiles import PROFILES, apply_profile
from sunsched.sim.simulator import simulate
from sunsched.sim.warmup import extend, scored_part
from sunsched.tests import test_simulator as TS

DT = 300.0


def _flat(soc, days=365, temp=25.0):
    n = int(days * 288)
    return np.full(n, soc), np.full(n, temp)


def test_lfp_matches_the_published_storage_point():
    soc, temp = _flat(0.5, 21241 / 24)
    fade = math.sqrt(float(np.sum(CH.lfp_calendar_k(soc, temp) ** 2) * DT))
    assert abs((1 - fade) - 0.9338) < 0.001


def test_calendar_life_ratios_at_25c():
    cfg = AgingCfg()
    r = {}
    for c, f in (("lmo", lambda s, t: aging_summary(s, t, DT, 365, cfg, 10)["battery_years_to_eol"]),
                 ("lfp", lambda s, t: CH.life_lfp(s, t, DT, 365)["life"]),
                 ("nmc", lambda s, t: CH.life_nmc(s, t, DT, 365)["life"])):
        r[c] = f(*_flat(0.5)) / f(*_flat(0.95)), f(*_flat(0.3)) / f(*_flat(0.95))
    assert abs(r["lmo"][0] - 1.60) < 0.01 and abs(r["lmo"][1] - 1.97) < 0.01
    assert abs(r["lfp"][0] - 2.05) < 0.01 and abs(r["lfp"][1] - 2.22) < 0.01
    assert abs(r["nmc"][0] - 2.30) < 0.01 and abs(r["nmc"][1] - 3.03) < 0.01


def test_chemistries_age_faster_when_hot_full_and_cycled():
    for f in (CH.life_lfp, CH.life_nmc):
        base = f(*_flat(0.6), DT, 365)["life"]
        assert f(*_flat(0.9), DT, 365)["life"] < base and f(*_flat(0.6, temp=40.0), DT, 365)["life"] < base
        n = 365 * 288
        wave = 0.6 + 0.2 * np.sin(2 * np.pi * np.arange(n) / 288)
        c = f(wave, np.full(n, 25.0), DT, 365)
        assert c["life"] < base and 0 < c["calendar_share"] < 1


def test_chemistries_off_by_default():
    soc, temp = _flat(0.5, 2)
    assert CH.chemistry_lives(soc, temp, DT, 2, AgingCfg(), 10, AgingCfg().chemistries) == {}


def test_profiles_break_even_and_slot_resolution():
    assert [PROFILES[p].break_even_s() for p in ("P2", "P3")] == [70.0, 58.0]
    assert abs(PROFILES["P4"].break_even_s() - 10.0) < 1e-9 and PROFILES["P1"].break_even_s() is None
    assert PROFILES["P1"].batch_size() == 24 and PROFILES["P4"].batch_size() == 5
    # Across one 5-minute slot, suspend never beats powering off, so the plan's
    # B* = 3 (P2) and 8 (P3), which assume a suspended processor, do not apply.
    for p in PROFILES.values():
        assert p.restart_j(300.0) == p.e_wake_cold_j
    assert PROFILES["P2"].batch_size(300.0) == 24 and PROFILES["P3"].batch_size(300.0) == 42
    for name, p in PROFILES.items():
        cfg = apply_profile(Config(), name)
        e = build_node_energy(cfg.node, {"triage": 10 ** 8, "full": 10 ** 9}, 300.0)
        assert (e.wake_j, e.triage_j, e.refine_j["full"]) == (p.e_wake_cold_j, p.e_triage_j, p.e_frame_j)


def test_plan_theorem_1_is_false():
    """The plan's wake rule (B* frames or a deadline) exceeds its claimed ratio,
    and waiting for the oldest deadline never wakes more often. Kept as a test
    so the claim is not used."""
    D, p = 48 * 3600.0, PROFILES["P4"]
    e, E = p.e_frame_j, p.e_wake_cold_j
    a = np.linspace(0, 0.99 * D, 2000)
    plan = math.ceil(len(a) / p.batch_size()) * E + len(a) * e
    lazy = E + len(a) * e                        # one wake serves every frame in time
    assert plan / lazy > 2 - e / (E + e)


def _run(policy, warm_days, soc_init=0.8):
    cfg, oc, energy, slots, harvest, temp, night, elev, n = TS.world()
    cfg.battery.soc_init = soc_init
    from sunsched.env.forecast import ReserveForecaster
    from sunsched.env.solar import dawn_dusk
    from sunsched.experiment import make_policy
    w = warm_days * cfg.slots_per_day
    frames, k = np.arange(len(slots)), 0
    if w:
        x = extend(harvest, temp, night, elev, frames, slots, w)
        harvest, temp, night, elev, frames, slots, k = (x[a] for a in (
            "harvest", "temp", "night", "elev", "frame_idx", "slots", "n_warm_frames"))
    fc = ReserveForecaster(n + w, int(36 * 3600 / cfg.slot_seconds), cfg.control.alpha, cfg.control.aci_gamma,
                           30, cfg.battery.eta_charge, cfg.battery.eta_discharge)
    sr, ss = dawn_dusk(elev, cfg.slots_per_day)
    ctx = RunContext(cfg=cfg, energy=energy, forecaster=fc, slots_per_day=cfg.slots_per_day,
                     slot_seconds=cfg.slot_seconds,
                     deadline_slots=int(cfg.control.deadline_h * 3600 / cfg.slot_seconds),
                     sunrise=sr, sunset=ss, n_slots=n + w)
    res = simulate(ctx, make_policy(policy, cfg), frames, slots, oc, harvest, temp, night, score_from=w)
    return cfg, n, w, res, (scored_part(res, w, k) if w else res)


def test_warmup_scores_exactly_one_window():
    for pol in ("always_now", "sunsched3"):
        cfg, n, w, full, part = _run(pol, 5)
        _, _, _, plain, _ = _run(pol, 0)
        assert len(part.soc) == n and len(part.frames) == len(plain.frames)
        assert np.array_equal(part.frames, plain.frames) and np.array_equal(part.capture_slot, plain.capture_slot)
        assert np.all(part.final_slot >= part.capture_slot)


def test_warmup_energy_balance_and_start_independence():
    eta = Config().battery
    cfg, n, w, full, part = _run("always_now", 5)
    C = cfg.capacity_j
    sd = eta.self_discharge_per_month / (30 * 86400) * cfg.slot_seconds * C
    lhs = (part.soc[-1] - full.soc[w - 1]) * C
    rhs = eta.eta_charge * part.harvested_j - part.consumed_j - part.curtailed_j - n * sd
    assert full.dead_slots == 0 and abs(lhs - rhs) < 1e-9 * C     # the counters cover the scored part only
    a, b = _run("always_now", 20, 0.8)[4].soc[0], _run("always_now", 20, 0.3)[4].soc[0]
    c, d = _run("always_now", 0, 0.8)[4].soc[0], _run("always_now", 0, 0.3)[4].soc[0]
    assert abs(a - b) <= abs(c - d) + 1e-12


def _meta(n_cam=40, n=400, days=200, seed=0, empty=0.4, bad_time=0.0, species=14):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cam):
        t = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, days * 86400, n)), unit="s")
        t = t.to_series(index=range(n)).dt.floor("s")
        t[rng.random(n) < bad_time] = pd.NaT
        lab = np.where(rng.random(n) < empty, "empty", [f"sp{k}" for k in rng.integers(0, species, n)])
        rows.append(pd.DataFrame(dict(location=f"c{c}", t=t, label=lab)))
    d = pd.concat(rows, ignore_index=True)
    d["empty"] = d.label == "empty"
    d["animal"] = ~d["empty"]
    return d


def test_eligibility_gates():
    good = E.gates(_meta(50), 1.0, 1.0)[0]
    assert good["all_pass"] and good["eligible_cameras"] == 50
    for kw, gate in ((dict(n_cam=20), "g2_cameras"), (dict(n_cam=50, bad_time=0.1), "g1_time"),
                     (dict(n_cam=50, empty=0.0), "g3_empty"), (dict(n_cam=50, species=5), "g4_species"),
                     (dict(n_cam=50, days=20), "g2_cameras")):
        g = E.gates(_meta(**kw), 1.0, 1.0)[0]
        assert not g[gate] and not g["all_pass"], (kw, gate)
    assert not E.gates(_meta(50), 1.0, 0.9)[0]["g6_urls"]


def test_camera_window_and_split():
    sec = np.r_[np.arange(10), np.arange(1000, 1100)]
    assert E.best_window(sec)[0] == 1000
    d = _meta(50)
    _, cams, roles = E.gates(d, 1.0, 1.0)
    assert roles == E.gates(d, 1.0, 1.0)[2]                        # deterministic
    sets = [set(v) for v in roles.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert (len(roles["eval"]), len(roles["calib"]), len(roles["train"])) == (25, 5, 20)
    assert E.camera_table(_meta(2, n=6000, days=300)).frames.max() == 5000


def test_stabbing_is_the_minimum_number_of_wakes():
    from itertools import combinations
    from sunsched.eval.wakeshare import stabbing_wakes, wake_share
    rng = np.random.default_rng(3)
    for _ in range(60):
        s = np.sort(rng.integers(0, 14, rng.integers(1, 7)))
        D = int(rng.integers(1, 5))
        points = range(0, 14 + D + 1)
        best = next(k for k in range(1, len(s) + 1)
                    if any(all(any(a <= p <= a + D for p in c) for a in s) for c in combinations(points, k)))
        assert stabbing_wakes(s, D) == best
    w = wake_share(np.array([0, 0, 1, 2, 1000]), 576, 12.0, 100.0)
    assert (w["n_capture_slots"], w["n_stab"]) == (4, 2) and abs(w["wake_share"] - 0.24) < 1e-12


def test_tier_training_and_distillation_run():
    """The training loop with a tiny stand-in trunk (timm is not needed here)."""
    import io
    import torch
    from PIL import Image
    from sunsched.vision import tiers as TR

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c, self.b = torch.nn.Conv2d(3, 8, 3, 2), torch.nn.BatchNorm2d(8)

        def forward(self, x):
            z = self.b(self.c(x))
            return torch.cat([z.mean((2, 3)), z.amax((2, 3))], 1)

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 64)
    blobs = []
    for k in y:
        b = io.BytesIO()
        Image.fromarray((rng.integers(0, 60, (40, 56, 3)) + 150 * k).astype(np.uint8)).save(b, "JPEG")
        blobs.append(b.getvalue())
    dev, rows = torch.device("cpu"), np.arange(64)
    tr = Tiny()
    bn = tr.b.running_mean.clone()
    tr, hd = TR.finetune(tr, (32, 32), blobs, rows, y, 2, dev, epochs=4, batch=16, lr=1e-2, head_lr=1e-2, workers=0, log=lambda *_: 0)
    X, Z = TR.extract(tr, hd, (32, 32), blobs, rows, dev, workers=0)
    assert X.shape == (64, 16) and (Z.argmax(1) == y).mean() > 0.9 and torch.equal(bn, tr.b.running_mean)
    s, h2 = TR.finetune(Tiny(), (16, 16), blobs, rows, y, 2, dev, epochs=4, batch=16, lr=1e-2, head_lr=1e-2, teacher=Z, workers=0, log=lambda *_: 0)
    assert (TR.extract(s, h2, (16, 16), blobs, rows, dev, workers=0)[1].argmax(1) == y).mean() > 0.9
    conv = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1, bias=False))
    assert TR.count_macs(conv, 8, 8) == 8 * 8 * 3 * 4 * 9


def _v5(sun=0.05, days=40, **sw):
    """One v5 run on the synthetic world; returns (result, policy)."""
    from sunsched.env.forecast import ReserveForecaster
    from sunsched.env.solar import dawn_dusk
    from sunsched.policies.v5 import SunSchedV5
    cfg, oc, energy, slots, harvest, temp, night, elev, n = TS.world(sun=sun, days=days)
    pol = SunSchedV5(cfg.control, cfg.baselines, **sw)
    fc = ReserveForecaster(n, int(36 * 3600 / cfg.slot_seconds), cfg.control.alpha, cfg.control.aci_gamma, 30,
                           cfg.battery.eta_charge, cfg.battery.eta_discharge)
    sr, ss = dawn_dusk(elev, cfg.slots_per_day)
    ctx = RunContext(cfg=cfg, energy=energy, forecaster=fc, slots_per_day=cfg.slots_per_day,
                     slot_seconds=cfg.slot_seconds,
                     deadline_slots=int(cfg.control.deadline_h * 3600 / cfg.slot_seconds),
                     sunrise=sr, sunset=ss, n_slots=n)
    return simulate(ctx, pol, np.arange(len(slots)), slots, oc, harvest, temp, night), pol, ctx


def test_v5_reduces_to_v4_when_switched_off():
    from sunsched.tests.test_simulator import run
    for sun in (0.05, 0.3):
        a = run("v4_full_surv", sun=sun, days=40)[2]
        b = _v5(sun, backoff=False, late=False)[0]
        assert np.array_equal(a.soc, b.soc) and np.array_equal(a.final_label, b.final_label)


def test_v5_respects_deadlines_and_late_saves_wakes():
    for sun in (0.05, 0.3, 2.0):
        r, _, ctx = _v5(sun)
        later = r.route == 3
        assert np.all((r.final_slot - r.capture_slot)[later] <= ctx.deadline_slots) and np.all(r.final_slot >= r.capture_slot)
    assert _v5(0.3, late=True)[0].b_wakes <= _v5(0.3, late=False)[0].b_wakes


def test_v5_backoff_releases_and_doubles():
    from types import SimpleNamespace as NS
    _, pol, _ = _v5(0.3, days=10)
    pol._hold_on, pol._ceiling, pol._held_last, pol._short, pol._b, pol._hold_until = True, 0.5, 20, True, 14, -1
    import sunsched.policies.v4 as v4
    orig = v4.SunSchedV4._reservoir_hold
    v4.SunSchedV4._reservoir_hold = lambda self, obs, judge=False, curtailed=False: setattr(self, "_hold_on", True)
    try:
        pol._reservoir_hold(NS(day=21))
        assert not pol._hold_on and pol._ceiling == 1.0 and pol._hold_until == 35 and pol._b == 28
        pol._reservoir_hold(NS(day=34))
        assert not pol._hold_on
        pol._reservoir_hold(NS(day=35))
        assert pol._hold_on
        pol._short = True
        pol._reservoir_hold(NS(day=36))
        assert pol._hold_until == 64 and pol._b == 56
        pol._short = True                              # a shortage with no recent hold is not the hold's fault
        pol._reservoir_hold(NS(day=50))
        assert pol._hold_until == 64
    finally:
        v4.SunSchedV4._reservoir_hold = orig


def test_lazy_ceiling_and_new_policies_run():
    from sunsched.experiment import LAZY_CEILINGS, V5_POLICIES
    from sunsched.tests.test_simulator import run
    for p in V5_POLICIES + list(LAZY_CEILINGS):
        r = run(p, sun=0.3, days=12)[2]
        assert np.all(r.final_slot >= 0), p
    assert run("lazy_c50", sun=3.0)[2].soc[2000:].max() <= 0.8 + 1e-9


def test_hypervolume():
    from sunsched.eval.pareto import front, hypervolume
    assert hypervolume([(1, 1)]) == 1 and hypervolume([]) == 0
    assert abs(hypervolume([(0.5, 1.0), (1.0, 0.5)]) - 0.75) < 1e-12
    assert hypervolume([(0.5, 1.0), (1.0, 0.5), (0.4, 0.4)]) == hypervolume([(0.5, 1.0), (1.0, 0.5)])
    assert front([(1, 1), (1, 0.5), (0.5, 1)]) == [(1, 1)]
    rng = np.random.default_rng(0)
    p = rng.random((30, 2))
    g = (np.stack(np.meshgrid(*[(np.arange(400) + .5) / 400] * 2), -1)).reshape(-1, 2)
    mc = np.mean([(p >= q).all(1).any() for q in g])
    assert abs(hypervolume(p) - mc) < 0.01
