"""Metrics used throughout the evaluation protocol (report section 10)."""
from dataclasses import dataclass
import numpy as np


def useful_inferences_per_joule(n_correct: int, total_energy_j: float) -> float:
    return n_correct / max(total_energy_j, 1e-9)


def downtime_fraction(depleted: np.ndarray) -> float:
    return float(depleted.mean())


def equivalent_full_cycles_per_year(half_cycles, trace_days: float) -> float:
    total = 0.5 * sum(depth for _, _, depth in half_cycles)
    return total * (365.0 / max(trace_days, 1e-9))


def total_embodied_carbon_kg(carbon_cfg, battery_capacity_wh: float, n_replacements: int) -> float:
    panel = carbon_cfg.panel_kgco2e_per_wp * carbon_cfg.panel_wp
    battery = carbon_cfg.battery_kgco2e_per_kwh * (battery_capacity_wh / 1000.0) * max(n_replacements, 1)
    return panel + carbon_cfg.board_kgco2e + battery


def amortised_embodied_carbon_kg(carbon_cfg, battery_capacity_wh: float,
                                   replacements_continuous: float) -> float:
    """Same as `total_embodied_carbon_kg` but for a continuous, non-floored
    replacement *rate* (see `projected_replacements_continuous`) -- the
    battery term can be below one unit, matching how an LCA amortises a
    capital good's embodied carbon over the fraction of its life actually
    consumed, rather than counting whole physical swaps."""
    panel = carbon_cfg.panel_kgco2e_per_wp * carbon_cfg.panel_wp
    battery = carbon_cfg.battery_kgco2e_per_kwh * (battery_capacity_wh / 1000.0) * max(replacements_continuous, 0.0)
    return panel + carbon_cfg.board_kgco2e + battery


def replacements_from_wear(final_wear: float) -> int:
    """How many times the battery had to be physically swapped over the
    deployment, given cumulative wear (1.0 = one battery's worth of life
    consumed). Integer and floored at 1, matching the real-world
    "how many truck rolls" question -- used for the main comparison table.
    Over a short trace this rounds every policy up to the same "1", which
    hides small differences in wear; use `projected_replacements_continuous`
    where the *rate* of battery consumption, not the truck-roll count, is
    what's being compared (Experiment 3)."""
    return max(1, int(np.ceil(final_wear)))


def projected_replacements_continuous(final_wear: float, trace_days: float,
                                        deployment_years: float) -> float:
    """Amortised (fractional) battery consumption over a full deployment
    horizon, extrapolated linearly from the wear rate observed on the
    simulated trace. Unlike `replacements_from_wear`, this is not ceiling'd
    or floored at 1 -- it is a *rate*, appropriate for comparing two
    policies' relative battery cost the way an LCA amortises embodied carbon
    over a product's useful life, not a count of physical swaps. A short
    trace under-samples rare deep-discharge events, so this should be read
    as directional (which policy costs the battery more), not as a precise
    replacement count."""
    years_simulated = max(trace_days, 1e-9) / 365.0
    return final_wear * (deployment_years / years_simulated)


def carbon_normalised_task_utility(total_value: float, carbon_kg: float) -> float:
    return total_value / max(carbon_kg, 1e-9)


def battery_carbon_share(carbon_cfg, battery_capacity_wh: float,
                         replacements: float) -> float:
    """Fraction of lifecycle carbon that the control policy can actually move.

    This is the precondition for Claim 1, and it has to be checked before any
    carbon-normalized comparison is believed. Panel and board are fixed at
    deployment; the battery is the only component a policy consumes faster or
    slower. If the battery term is a rounding error against the board, then
    carbon-normalized task utility is total value divided by a constant, every
    policy ranks exactly as it does on raw value, and the carbon inversion
    cannot appear no matter how good the controller is.

    Measured on the repository's default constants: a 4 Wh battery against a
    22 kgCO2e board is 1.5% of lifecycle carbon at a *full* replacement, and
    0.06% at the wear the simulation actually produces. Claim 1 is untestable
    at that parameterization -- not false, unmeasurable.
    """
    battery = carbon_cfg.battery_kgco2e_per_kwh * (battery_capacity_wh / 1000.0) * max(replacements, 0.0)
    fixed = carbon_cfg.panel_kgco2e_per_wp * carbon_cfg.panel_wp + carbon_cfg.board_kgco2e
    return battery / max(battery + fixed, 1e-12)


def gco2e_per_correct_inference(carbon_kg: float, n_correct: int) -> float:
    """The paper's primary metric.

    Leading with joules per inference concedes the framing that energy is the
    objective. For an off-grid node operational carbon is approximately zero,
    so what a policy actually controls is embodied carbon via battery wear --
    which means the honest denominator is useful work and the honest numerator
    is grams of CO2e, not joules.
    """
    return 1000.0 * carbon_kg / max(n_correct, 1)


def percent_of_control_oracle(value: float, control_oracle_value: float) -> float:
    """Against the harvest-foresight oracle, which is the honest control ceiling.

    The original pipeline reported percent of a *clairvoyant* oracle whose
    reward used per-image ground-truth correctness, so it woke only on slots it
    already knew would be classified correctly. With peak accuracy at 47.6%,
    any non-clairvoyant policy is capped near that ratio by construction, and
    "76% of oracle" was close to a restatement of the model's accuracy rather
    than a measure of control quality. Report both, separately, and treat only
    this one as a control metric.
    """
    return 100.0 * value / max(control_oracle_value, 1e-9)


def irreducible_gap(control_oracle_value: float, clairvoyant_value: float) -> float:
    """The part of the gap to the clairvoyant oracle that no controller can
    close, because it comes from classifier uncertainty rather than from
    scheduling. Reported alongside, never folded into, the control metric."""
    return 100.0 * (1.0 - control_oracle_value / max(clairvoyant_value, 1e-9))


def value_of_forecast(ours_value: float, reactive_value: float, oracle_value: float) -> float:
    denom = oracle_value - reactive_value
    if abs(denom) < 1e-9:
        return float("nan")
    return (ours_value - reactive_value) / denom


def percent_of_oracle(value: float, oracle_value: float) -> float:
    return 100.0 * value / max(oracle_value, 1e-9)


@dataclass
class RunSummary:
    policy: str
    site: str
    value_recall: float
    event_recall: float
    total_value: float
    n_correct: int
    total_energy_j: float
    inferences_per_joule: float
    downtime_frac: float
    n_events: int
    n_events_captured: int
    equiv_full_cycles_per_year: float
    final_wear: float
    n_replacements: int
    carbon_kg: float
    ctu: float
    controller_overhead_j: float = 0.0

    def as_row(self) -> dict:
        return self.__dict__.copy()
