"""End-to-end invariants of the simulator on a small synthetic world.

No downloads, no trained model: outcomes, captures and weather are generated
here, so this runs in seconds on any machine.
"""
import numpy as np

from sunsched.config import Config
from sunsched.env.forecast import ReserveForecaster
from sunsched.env.solar import dawn_dusk
from sunsched.experiment import ABLATIONS, BASELINES, OURS, make_policy
from sunsched.policies.base import RunContext
from sunsched.sim.node import build_node_energy
from sunsched.sim.outcomes import Outcomes
from sunsched.sim.simulator import (DROPPED, REFINED_LATER, REFINED_NOW, UNCLASSIFIED,
                                    simulate)

POLICIES = BASELINES + OURS + ABLATIONS


def world(days=20, n_frames=600, seed=0, sun=1.0, capacity_wh=10.0):
    rng = np.random.default_rng(seed)
    cfg = Config()
    cfg.battery.capacity_wh = capacity_wh
    spd = cfg.slots_per_day
    n = days * spd
    names = ["bird", "coyote", "empty"]
    labels = rng.integers(0, 3, n_frames)

    def preds(acc):
        p = labels.copy()
        wrong = rng.random(n_frames) > acc
        p[wrong] = rng.integers(0, 3, wrong.sum())
        return p

    # An informative detector: animals score high, empties low.
    p_animal = np.clip(np.where(labels == 2, 0.2, 0.8) + rng.normal(0, 0.15, n_frames), 0, 1)
    oc = Outcomes(class_names=names, labels=labels, values=np.where(labels == 2, 0.1, 1.0),
                  locations=np.array(["1"] * n_frames),
                  pred={"triage": preds(0.5), "lite": preds(0.7), "full": preds(0.85)},
                  conf={k: rng.random(n_frames) for k in ("triage", "lite", "full")},
                  p_animal=p_animal,
                  macs={"triage": 40_000_000, "lite": 300_000_000, "full": 600_000_000},
                  gain_edges=np.array([0.0, 0.33, 0.66, 1.0 + 1e-9]),
                  gain_tables={"lite": np.array([0.0, 0.05, 0.2]), "full": np.array([0.0, 0.08, 0.3])},
                  empty_idx=2)
    hour = (np.arange(n) % spd) * cfg.solar.slot_minutes / 60.0
    elev = 60.0 * np.sin(np.pi * (hour - 6.0) / 12.0)
    irr = np.where(elev > 0, 900.0 * np.sin(np.deg2rad(np.clip(elev, 0, 90))), 0.0)
    irr *= np.repeat(rng.uniform(0.2, 1.0, days), spd)
    night_slots, day_slots = np.where(elev < 0)[0], np.where(elev >= 0)[0]
    slots = np.sort(np.concatenate([rng.choice(night_slots, int(0.6 * n_frames)),
                                    rng.choice(day_slots, n_frames - int(0.6 * n_frames))]))
    energy = build_node_energy(cfg.node, oc.macs, cfg.slot_seconds)
    harvest = irr * sun * 0.004 * cfg.solar.panel_efficiency * cfg.solar.mppt_efficiency * cfg.slot_seconds
    return cfg, oc, energy, slots, harvest, np.full(n, 25.0), elev < 0, elev, n


def run(policy_name, soc_start=None, **kw):
    cfg, oc, energy, slots, harvest, temp, night, elev, n = world(**kw)
    policy = make_policy(policy_name, cfg)
    fc = ReserveForecaster(n, int(36 * 3600 / cfg.slot_seconds), cfg.control.alpha,
                           cfg.control.aci_gamma, 30, cfg.battery.eta_charge, cfg.battery.eta_discharge,
                           conformal=getattr(policy, "uses_conformal", True))
    sunrise, sunset = dawn_dusk(elev, cfg.slots_per_day)
    ctx = RunContext(cfg=cfg, energy=energy, forecaster=fc, slots_per_day=cfg.slots_per_day,
                     slot_seconds=cfg.slot_seconds,
                     deadline_slots=int(cfg.control.deadline_h * 3600 / cfg.slot_seconds),
                     sunrise=sunrise, sunset=sunset, n_slots=n)
    res = simulate(ctx, policy, np.arange(len(slots)), slots, oc, harvest, temp, night, soc_start=soc_start)
    return cfg, ctx, res, harvest


def test_every_policy_runs_and_finalises_every_frame():
    for name in POLICIES:
        cfg, ctx, res, _ = run(name)
        assert np.all(res.final_slot >= 0), f"{name}: a frame never got a final label"
        assert np.all(res.final_slot >= res.capture_slot), f"{name}: label before capture"
        labelled = res.final_label >= 0
        assert np.all(labelled | np.isin(res.route, [DROPPED, UNCLASSIFIED])), name
        assert np.all((res.soc >= -1e-9) & (res.soc <= 1 + 1e-9)), name


def test_policy_signatures():
    _, _, r, _ = run("always_now")
    assert r.deferred == 0 and not np.any(r.route == REFINED_LATER)
    _, _, r, _ = run("triage_only")
    assert r.b_wakes == 0 and not np.any(np.isin(r.route, [REFINED_NOW, REFINED_LATER]))
    _, _, r, _ = run("sunsched_no_gate")
    assert r.deferred > 0 and not np.any(r.route == REFINED_NOW), "no_gate must always defer"
    _, _, r, _ = run("sunsched_no_defer")
    assert r.deferred == 0, "no_defer must never queue"


def test_gate_refines_now_with_a_large_battery():
    """Plenty of stored energy above the reserve: the gate should classify on
    arrival rather than make frames wait."""
    _, _, r, _ = run("sunsched", sun=3.0, capacity_wh=40.0, soc_start=0.9)
    now, later = np.mean(r.route == REFINED_NOW), np.mean(r.route == REFINED_LATER)
    assert now > later, f"large battery: refined now {now:.2f} vs later {later:.2f}"


def test_gate_defers_at_night_with_a_small_battery():
    """A battery holding about one night of essential energy has no surplus
    after dark, so night captures worth refining must wait for the sun."""
    cfg, ctx, r, _ = run("sunsched", sun=3.0, capacity_wh=0.08, soc_start=0.5)
    assert r.deferred > 0, "small battery: nothing was deferred"


def test_small_battery_still_refines_in_daylight():
    """Regression test. With a battery smaller than one night's reserve, v2
    used to see no surplus ever, so every deferred frame expired unrefined. The
    rest of today's harvest must count as surplus before sunset."""
    _, _, r, _ = run("sunsched", sun=3.0, capacity_wh=0.08, soc_start=0.5)
    assert np.mean(r.route == REFINED_LATER) > 0.05, "deferred frames are never refined"


def test_empties_are_not_refined_when_the_detector_is_confident():
    """Gain is zero in the lowest p_animal bin, so those frames stay at triage."""
    cfg, oc, *_ = world()
    _, _, r, _ = run("sunsched", sun=3.0, capacity_wh=40.0, soc_start=0.9)
    low = oc.p_animal[r.frames] < 0.33
    assert not np.any(np.isin(r.route[low], [REFINED_NOW, REFINED_LATER]))


def test_deadline_is_respected():
    for name in ("sunsched", "sunsched_no_gate", "lazy_defer"):
        cfg, ctx, res, _ = run(name)
        later = res.route == REFINED_LATER
        assert np.all((res.final_slot - res.capture_slot)[later] <= ctx.deadline_slots), name


def test_energy_is_conserved():
    checked = 0
    for name in ("always_now", "sunsched", "lazy_defer", "sunsched_no_gate"):
        cfg, ctx, res, harvest = run(name, sun=3.0)
        if res.dead_slots:
            continue                    # clamping at zero breaks the identity by design
        checked += 1
        C = cfg.capacity_j
        sd = cfg.battery.self_discharge_per_month / (30 * 86400) * ctx.slot_seconds * C
        start = cfg.battery.soc_init * C
        expected = (start + cfg.battery.eta_charge * harvest.sum() - res.consumed_j
                    - res.curtailed_j - sd * ctx.n_slots)
        assert abs(expected - res.soc[-1] * C) < 1e-6 * C, f"{name}: energy not conserved"
    assert checked >= 2, "too few runs stayed alive to check conservation"


def test_soc_start_is_used():
    _, _, lo, _ = run("triage_only", soc_start=0.2, days=2)
    _, _, hi, _ = run("triage_only", soc_start=0.9, days=2)
    assert hi.soc[0] > lo.soc[0]


def test_charge_never_exceeds_ceiling():
    cfg, ctx, res, _ = run("sunsched", sun=3.0)
    prev = cfg.battery.soc_init
    for s, c in zip(res.soc, res.ceiling):
        assert s <= max(c, prev) + 1e-9, "charged above the ceiling"
        prev = s


def test_ceiling_lowers_state_of_charge():
    _, _, with_ceiling, _ = run("sunsched", sun=3.0, days=40)
    _, _, without, _ = run("sunsched_no_ceiling", sun=3.0, days=40)
    assert with_ceiling.soc.mean() < without.soc.mean()
