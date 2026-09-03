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


def replacements_from_wear(final_wear: float) -> int:
    """How many times the battery had to be replaced over the deployment,
    given cumulative wear (1.0 = one battery's worth of life consumed)."""
    return max(1, int(np.ceil(final_wear)))


def carbon_normalised_task_utility(total_value: float, carbon_kg: float) -> float:
    return total_value / max(carbon_kg, 1e-9)


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
