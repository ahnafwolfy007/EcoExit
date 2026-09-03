"""Central configuration for the EcoExit pipeline.

Every constant a reviewer might question lives here, with a source note.
Constants marked LITERATURE must be replaced with your own measurements or a
cited datasheet before any number derived from them is claimed as a result.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict
import json


@dataclass
class ModelCfg:
    """Elastic backbone: k resolutions x m exits from one set of weights."""
    resolutions: Tuple[int, ...] = (16, 24, 32)   # ePerceptive: keep them well separated
    n_exits: int = 3
    base_width: int = 16
    blocks_per_stage: int = 2
    n_classes: int = 100
    exit_loss_weights: Tuple[float, ...] = (0.4, 0.7, 1.0)


@dataclass
class TrainCfg:
    epochs: int = 12
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    sandwich_n_random: int = 1    # sandwich rule: smallest + largest + n random
    train_subset: int = 0         # 0 = use all
    val_fraction: float = 0.1
    seed: int = 0
    num_workers: int = 0          # 0 is safest on Windows


@dataclass
class EnergyCfg:
    """Decomposed energy model.

    E(r,k) = b0 + b1*MACs + b2*ActBytes + b3*WeightBytes + P_static * T(r,k)

    MACs, ActBytes, WeightBytes and T are MEASURED from the real model on the
    real machine (scripts/02_profile_energy.py). The coefficients and power
    rails below are LITERATURE defaults for a Jetson-Nano-class node; replace
    them with an INA219/INA3221 fit for any claim about real joules.
    """
    b1_pj_per_mac: float = 3.7
    b2_pj_per_act_byte: float = 62.0
    b3_pj_per_weight_byte: float = 62.0
    b0_j: float = 2.0e-4
    p_static_w: float = 1.9          # board rail draw DURING the brief active/compute window
    p_idle_w: float = 0.15           # deep-sleep / power-gated draw BETWEEN wake windows (~150 mW
    # MCU+RTC keeping the schedule and gating power to the main SBC). This is why the duty knob
    # exists at all -- a Jetson left merely "idle but powered" draws ~1.5-3 W continuously
    # (Hole H3) and would drain a small battery in hours regardless of policy; a real solar
    # node instead power-gates the SBC between wake windows.
    e_sense_j_per_kpix: float = 1.1e-3
    e_pre_j_per_kpix: float = 2.4e-4
    # LITERATURE: waking a power-gated SBC is a suspend/resume-plus-camera-init cycle
    # (seconds, several watts), not an instant clock tick -- this is what makes *how
    # often* to wake (Contribution E) as important a lever as *how* to infer.
    e_wake_j: float = 3.0
    measurement_noise_frac: float = 0.03
    # The elastic backbone here is intentionally small (CPU-trainable in minutes), so its
    # raw measured per-inference energy (used as-is for the Contribution D mismatch analysis
    # in scripts/02) is far below what a production-scale vision backbone at full resolution
    # would draw. `system_scale_factor` rescales ONLY the per-inference/sense terms when
    # building the energy table the closed-loop system simulation (scripts/04) uses, so the
    # duty/resolution/exit trade-off happens at a believable absolute wattage. It never
    # touches the raw MACs/bytes/time numbers or the MACs-only-vs-decomposed comparison.
    system_scale_factor: float = 250.0


@dataclass
class SolarCfg:
    # Deliberately undersized relative to a 60s-cadence Jetson-class workload: the
    # point of this project is a controller for a genuinely energy-scarce deployment.
    # A generously provisioned panel would make adaptive control largely unnecessary.
    panel_area_m2: float = 0.008
    panel_efficiency: float = 0.19
    slot_seconds: int = 60
    plan_block_slots: int = 15
    days: int = 45          # 45 daily blocks gives conformal coverage a real sample size
    seed: int = 0


@dataclass
class BatteryCfg:
    capacity_wh: float = 4.0
    soc_init_frac: float = 0.6
    soc_min_frac: float = 0.05
    eta_charge: float = 0.92
    eta_discharge: float = 0.96
    self_discharge_per_month: float = 0.025
    dod_points: Tuple[float, ...] = (0.2, 0.5, 0.8, 1.0)
    cycles_at_dod: Tuple[float, ...] = (12000.0, 6000.0, 3000.0, 2000.0)


@dataclass
class CarbonCfg:
    """Embodied carbon. All LITERATURE; sweep them, never claim a point value."""
    panel_kgco2e_per_wp: float = 0.55
    panel_wp: float = 1.5    # matches SolarCfg's 0.008 m^2 x 0.19 eff x 1000 W/m^2
    board_kgco2e: float = 22.0
    battery_kgco2e_per_kwh: float = 85.0
    deployment_years: float = 5.0


@dataclass
class StreamCfg:
    events_per_day: float = 24.0
    burst_size_mean: float = 3.0
    crepuscular_boost: float = 3.0
    background_value: float = 0.02
    event_value: float = 1.0
    seed: int = 0


@dataclass
class ControlCfg:
    alpha: float = 0.10
    aci_gamma: float = 0.02
    horizon_hours: float = 24.0
    replan_minutes: float = 30.0
    soc_bins: int = 41        # realistic tabular-controller resolution (LUT, Q-learning, MDP)
    oracle_soc_bins: int = 201  # finer grid for the offline DP ceiling -- an upper bound
    # should not itself be handicapped by the same coarse discretisation a cheap embedded
    # controller would use; 201 bins keeps quantisation error well under one slot's energy
    # delta over the full SoC range.


SITES: Dict[str, Dict[str, float]] = {
    "dhaka":   dict(lat=23.8, kc_mean=0.62, kc_rho=0.93, day_of_year=196),
    "nairobi": dict(lat=-1.3, kc_mean=0.66, kc_rho=0.90, day_of_year=196),
    "munich":  dict(lat=48.1, kc_mean=0.52, kc_rho=0.95, day_of_year=196),
    "phoenix": dict(lat=33.4, kc_mean=0.83, kc_rho=0.88, day_of_year=196),
    "bergen":  dict(lat=60.4, kc_mean=0.41, kc_rho=0.96, day_of_year=196),
}
TRAIN_SITE = "dhaka"
TEST_SITES = ["nairobi", "munich", "phoenix", "bergen"]


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    energy: EnergyCfg = field(default_factory=EnergyCfg)
    solar: SolarCfg = field(default_factory=SolarCfg)
    battery: BatteryCfg = field(default_factory=BatteryCfg)
    carbon: CarbonCfg = field(default_factory=CarbonCfg)
    stream: StreamCfg = field(default_factory=StreamCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    dataset: str = "cifar100"
    out_dir: str = "results"
    artifacts_dir: str = "artifacts"

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @property
    def battery_capacity_j(self) -> float:
        return self.battery.capacity_wh * 3600.0


def quick(cfg: Config) -> Config:
    """Laptop-friendly preset: minutes, not hours."""
    cfg.train.epochs = 3
    cfg.train.train_subset = 8000
    cfg.train.sandwich_n_random = 0
    cfg.model.base_width = 12
    cfg.solar.days = 7
    return cfg
