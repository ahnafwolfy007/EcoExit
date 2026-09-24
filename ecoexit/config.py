"""Central configuration for the EcoExit pipeline.

Every constant a reviewer might question lives here, with a source note.
Constants marked LITERATURE must be replaced with measurements or a cited
datasheet before any number derived from them is claimed as a result. The
sensitivity sweep in scripts/04 exists because several of them currently
determine the results more than the method does.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Optional
import json


@dataclass
class ModelCfg:
    """Elastic backbone: k resolutions x m exits from one set of weights."""
    resolutions: Tuple[int, ...] = (16, 24, 32)
    n_exits: int = 3
    base_width: int = 16
    blocks_per_stage: int = 2
    n_classes: int = 100
    exit_loss_weights: Tuple[float, ...] = (0.4, 0.7, 1.0)
    # Phase 3: a named torchvision/timm backbone with exit heads grafted on at
    # matched depths. None keeps the bespoke ElasticNet, which is fine for
    # Phase 0 plumbing but is not comparable to published work.
    backbone: Optional[str] = None


@dataclass
class TrainCfg:
    epochs: int = 12
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    sandwich_n_random: int = 1
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
    rails are LITERATURE defaults for a Jetson-Nano-class node.
    """
    b1_pj_per_mac: float = 3.7
    b2_pj_per_act_byte: float = 62.0
    b3_pj_per_weight_byte: float = 62.0
    b0_j: float = 2.0e-4
    p_static_w: float = 1.9          # rail draw DURING the active/compute window
    p_idle_w: float = 0.15           # deep-sleep draw BETWEEN wake windows
    e_sense_j_per_kpix: float = 1.1e-3
    e_pre_j_per_kpix: float = 2.4e-4
    e_wake_j: float = 3.0            # suspend/resume plus camera init
    measurement_noise_frac: float = 0.03
    # Rescales ONLY the per-inference/sense terms when building the table the
    # closed-loop simulation uses, so the trade-off happens at a believable
    # absolute wattage for a production-scale backbone. Never touches the raw
    # MACs/bytes/time numbers or the MACs-only-vs-decomposed comparison.
    # This is a free parameter and must be swept, not asserted.
    system_scale_factor: float = 250.0


@dataclass
class SolarCfg:
    panel_area_m2: float = 0.008
    panel_efficiency: float = 0.19
    slot_seconds: int = 60
    plan_block_slots: int = 15
    days: int = 45
    seed: int = 0
    # "analytic" = Haurwitz clear sky x AR(1) cloud process, fully offline.
    # "pvgis"    = real measured hourly irradiance at each site's coordinates.
    source: str = "analytic"
    pvgis_year: int = 2020


@dataclass
class BatteryCfg:
    capacity_wh: float = 4.0
    soc_init_frac: float = 0.6
    soc_min_frac: float = 0.05
    eta_charge: float = 0.92
    eta_discharge: float = 0.96
    self_discharge_per_month: float = 0.025
    # LITERATURE datasheet curve; CycleLifeCurve.fit turns it into N_f(d)=n0*d^-k.
    # Replace with a fit to the Severson et al. cells for a measured curve.
    dod_points: Tuple[float, ...] = (0.2, 0.5, 0.8, 1.0)
    cycles_at_dod: Tuple[float, ...] = (12000.0, 6000.0, 3000.0, 2000.0)


@dataclass
class WearCfg:
    """The path-dependent wear price (Claim 2)."""
    mode: str = "path"                 # "none" | "proxy" | "path"
    kappa_scale: float = 1.0           # multiplier on the carbon-derived kappa
    proxy_nominal_depth: float = 0.30  # depth the energy-proportional strawman is frozen at
    min_swing: float = 1e-4            # rainflow dither floor


@dataclass
class CarbonCfg:
    """Embodied carbon. All LITERATURE; sweep them, never claim a point value."""
    panel_kgco2e_per_wp: float = 0.55
    panel_wp: float = 1.5
    board_kgco2e: float = 22.0
    battery_kgco2e_per_kwh: float = 85.0
    deployment_years: float = 5.0
    profile: str = "sbc"


# Node hardware profiles. The carbon claim is only testable on a node where the
# battery is a material share of embodied carbon -- see
# `ecoexit.eval.metrics.battery_carbon_share`. The default SBC profile is NOT
# such a node: a 22 kgCO2e board dwarfs a 4 Wh battery, so every policy scores
# the same carbon and the inversion cannot appear. Kept as the default only
# because it is what the original pipeline used; use "mcu" for the carbon work.
CARBON_PROFILES: Dict[str, Dict[str, float]] = {
    # Jetson-class single-board computer. Battery term is immaterial.
    "sbc": dict(board_kgco2e=22.0, panel_wp=1.5, battery_capacity_wh=4.0),
    # MCU-class camera-trap node: small PCB, microcontroller, camera module,
    # and the kind of pack such deployments actually carry.
    "mcu": dict(board_kgco2e=2.5, panel_wp=1.5, battery_capacity_wh=8.0),
    # Same board, a pack sized so wear accumulates over a 5-year deployment.
    "mcu_small_battery": dict(board_kgco2e=2.5, panel_wp=1.5, battery_capacity_wh=3.0),
}


def apply_carbon_profile(cfg: "Config", name: str) -> "Config":
    """Switch node hardware class. Changes both carbon accounting and physics,
    because battery capacity sets how deeply a given draw cycles the pack."""
    if name not in CARBON_PROFILES:
        raise KeyError(f"unknown carbon profile {name!r}; have {sorted(CARBON_PROFILES)}")
    p = CARBON_PROFILES[name]
    cfg.carbon.board_kgco2e = p["board_kgco2e"]
    cfg.carbon.panel_wp = p["panel_wp"]
    cfg.battery.capacity_wh = p["battery_capacity_wh"]
    cfg.carbon.profile = name
    return cfg


@dataclass
class StreamCfg:
    events_per_day: float = 24.0
    burst_size_mean: float = 3.0
    crepuscular_boost: float = 3.0
    background_value: float = 0.02
    event_value: float = 1.0
    seed: int = 0
    # "synthetic" = Poisson-cluster crepuscular process (Phase 0 default).
    # "camera_trap" = real capture timestamps from camera-trap metadata.
    source: str = "synthetic"


@dataclass
class ControlCfg:
    alpha: float = 0.10
    aci_gamma: float = 0.02
    horizon_hours: float = 24.0    # horizon the stored energy is amortized over
    replan_minutes: float = 30.0
    reserve_frac: float = 0.10     # buffer above the hard floor the planner keeps
    trust: float = 1.0             # 1 = trust the conformal bound, 0 = full reserve
    soc_bins: int = 41
    oracle_soc_bins: int = 201


@dataclass
class DataCfg:
    """Where the downloaded datasets live. scripts/00_fetch_datasets.py fills
    this directory; DATASETS.md lists every source and link."""
    root: str = "./data"
    cifar_dir: str = "./data/cifar-100-python"
    camera_trap_dir: str = "./data/camera_traps"
    pvgis_cache_dir: str = "./data/pvgis"
    battery_dir: str = "./data/battery"
    # Which camera-trap corpus supplies event timing.
    camera_trap_source: str = "caltech"      # "caltech" | "serengeti"


@dataclass
class ExperimentCfg:
    """Statistical and sweep protocol."""
    n_seeds: int = 5
    ci_level: float = 0.95
    n_bootstrap: int = 2000
    # The regime coordinate every headline figure is plotted against.
    target_ratio: float = 0.75
    ratio_sweep: Tuple[float, ...] = (0.4, 0.55, 0.7, 0.85, 1.0, 1.3)


SITES: Dict[str, Dict[str, float]] = {
    "dhaka":   dict(lat=23.8103, lon=90.4125, kc_mean=0.62, kc_rho=0.93, day_of_year=196),
    "nairobi": dict(lat=-1.2921, lon=36.8219, kc_mean=0.66, kc_rho=0.90, day_of_year=196),
    "munich":  dict(lat=48.1351, lon=11.5820, kc_mean=0.52, kc_rho=0.95, day_of_year=196),
    "phoenix": dict(lat=33.4484, lon=-112.0740, kc_mean=0.83, kc_rho=0.88, day_of_year=196),
    "bergen":  dict(lat=60.3913, lon=5.3221, kc_mean=0.41, kc_rho=0.96, day_of_year=196),
    # Co-located with the camera-trap corpora, for the Claim 4 benchmark.
    "serengeti": dict(lat=-2.3333, lon=34.8333, kc_mean=0.64, kc_rho=0.91, day_of_year=196),
    "caltech":   dict(lat=32.6000, lon=-116.8000, kc_mean=0.78, kc_rho=0.89, day_of_year=196),
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
    wear: WearCfg = field(default_factory=WearCfg)
    carbon: CarbonCfg = field(default_factory=CarbonCfg)
    stream: StreamCfg = field(default_factory=StreamCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    data: DataCfg = field(default_factory=DataCfg)
    experiment: ExperimentCfg = field(default_factory=ExperimentCfg)
    dataset: str = "cifar100"
    out_dir: str = "results"
    artifacts_dir: str = "artifacts"

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @property
    def battery_capacity_j(self) -> float:
        return self.battery.capacity_wh * 3600.0

    @property
    def slots_per_day(self) -> int:
        return int(round(86400 / self.solar.slot_seconds))

    @property
    def replan_slots(self) -> int:
        return max(int(round(self.control.replan_minutes * 60 / self.solar.slot_seconds)), 1)


def quick(cfg: Config) -> Config:
    """Laptop-friendly preset: minutes, not hours."""
    cfg.train.epochs = 3
    cfg.train.train_subset = 8000
    cfg.train.sandwich_n_random = 0
    cfg.model.base_width = 12
    cfg.solar.days = 7
    cfg.experiment.n_seeds = 2
    cfg.experiment.ratio_sweep = (0.55, 0.85)
    return cfg
