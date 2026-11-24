"""Every constant the results depend on, in one place.

Constants marked ASSUMPTION have no measurement behind them in this project
(there is no hardware). Each one is swept in pipeline/05_sweeps.py, and a
conclusion only counts if it holds across its sweep range.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple
import json


@dataclass
class DataCfg:
    # "cct20" is the development corpus; "serengeti" is the held-out test
    # corpus of PREREGISTRATION.md (ported from the v2 branch for the paper
    # revision). `apply_corpus` sets the fields that depend on it.
    corpus: str = "cct20"
    root: str = "./data"
    cct20_dir: str = "./data/cct20"
    images_archive: str = "./data/cct20/eccv_18_all_images_sm.tar.gz"
    # If the archive has already been extracted, point this at the folder and
    # the extractor reads files from disk instead of streaming the archive.
    images_dir: str = ""
    annotations_dir: str = "./data/cct20/eccv_18_annotation_files"
    pvgis_dir: str = "./data/pvgis"
    # CCT20 ("Recognition in Terra Incognita", ECCV 2018) splits. The "cis"
    # splits share the 10 training camera locations; "trans" splits are
    # different locations. Heads are fit on cis locations only, calibrated on
    # trans_val (1 new location), and every simulated deployment replays
    # trans_test (9 further new locations), so no simulated camera was seen in
    # training. Never replace these with a random split: a camera trap's
    # background is fixed, and a random split rewards memorising it.
    train_splits: Tuple[str, ...] = ("train", "cis_val", "cis_test")
    calib_split: str = "trans_val"
    eval_split: str = "trans_test"
    # Snapshot Serengeti (test corpus). Images are fetched one by one from LILA,
    # downscaled on arrival and cached small. The selection rule is fixed in
    # sunsched/data/serengeti.py and PREREGISTRATION.md; these values are the
    # ones committed there before any Serengeti data was downloaded.
    serengeti_dir: str = "./data/serengeti"
    serengeti_season: str = "S10"
    serengeti_train_images: int = 10000
    serengeti_train_empty_frac: float = 0.2
    serengeti_calib_cameras: int = 3
    serengeti_eval_budget: int = 15000
    serengeti_min_eval_cameras: int = 6
    # PREREGISTRATION_v2.md test B: a later Serengeti season, excluding every
    # camera the S10 test used (its eval and calibration cameras).
    serengeti_exclude_season: str = ""
    # PREREGISTRATION_v2.md test A: Caltech Camera Traps cameras outside CCT20.
    cct_new_dir: str = "./data/cct_new"
    cct_new_train_images: int = 10000
    cct_new_train_empty_frac: float = 0.2
    cct_new_calib_cameras: int = 3
    cct_new_eval_budget: int = 20000
    # Ten cameras, not six: with the camera as the unit of analysis, an exact
    # Wilcoxon test on six camera means cannot go below p = 0.031 (two-sided).
    # Raised before any cct_new or S11 data was downloaded.
    cct_new_min_eval_cameras: int = 10
    # PREREGISTRATION_v3.md test corpora (sunsched/data/lila.py).
    lila_train_images: int = 10000
    lila_train_empty_frac: float = 0.2


@dataclass
class VisionCfg:
    # Pretrained ImageNet trunk, frozen. Only small exit heads are trained, so
    # the whole vision stage runs on CPU by default.
    backbone: str = "mobilenet_v3_large"
    # "cpu", "cuda" or "auto", set by --device. Feature extraction and head
    # training only, and it changes speed, not results: see
    # sunsched/vision/device.py for why the device cannot move a number here.
    device: str = "cpu"
    # Indices into torchvision's mobilenet_v3_large().features. 6 ends the
    # 40-channel stage, 12 the 112-channel stage, 16 is the final 960-channel conv.
    taps: Tuple[int, ...] = (6, 12, 16)
    # Input heights; width keeps the CCT20 aspect ratio (1024x747).
    resolutions: Tuple[int, ...] = (160, 320)
    aspect: float = 1024 / 747
    batch_size: int = 32
    head_hidden: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    seed: int = 0
    # Operating points as (resolution index, exit index).
    triage_op: Tuple[int, int] = (0, 0)
    refine_ops: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {"lite": (1, 1), "full": (1, 2)})
    n_conf_bins: int = 10
    gain_prior_count: float = 20.0      # shrinkage of per-bin gain estimates


@dataclass
class TaskCfg:
    """Value of a correctly classified frame, by its true class."""
    value_animal: float = 1.0
    value_vehicle: float = 0.3
    value_empty: float = 0.1
    vehicle_classes: Tuple[str, ...] = ("car",)
    empty_class: str = "empty"


@dataclass
class NodeCfg:
    """Two-tier node, all ASSUMPTION (datasheet-typical ranges, swept).

    Tier A is an always-available low-power path (camera controller plus a
    small NPU-class accelerator) that captures and triages every frame.
    Tier B is an application processor (Raspberry Pi / Jetson class) that must
    be woken to run the deep exits. Waking it is the dominant cost, which is
    why batching refinements into one wake matters.
    """
    p_sleep_w: float = 0.002                 # PIR armed, RTC, MCU deep sleep
    e_capture_day_j: float = 0.4             # PIR wake + sensor + exposure
    e_capture_night_j: float = 1.2           # plus IR illuminator
    tier_a_pj_per_mac: float = 20.0
    tier_a_overhead_j: float = 0.05
    e_store_j: float = 0.02                  # write frame to flash for later
    tier_b_wake_j: float = 12.0              # resume, init, tail energy
    tier_b_active_w: float = 3.0
    tier_b_gmac_per_s: float = 4.0           # effective throughput
    tier_b_per_frame_j: float = 0.05         # load frame from flash, bookkeeping
    queue_capacity: int = 20000              # frames that fit on flash


@dataclass
class BatteryCfg:
    capacity_wh: float = 10.0
    soc_init: float = 0.8
    soc_floor: float = 0.05
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    self_discharge_per_month: float = 0.02


@dataclass
class AgingCfg:
    """Semi-empirical Li-ion degradation model of Xu et al., "Modeling of
    Lithium-Ion Battery Degradation for Cell Life Assessment", IEEE Trans.
    Smart Grid, 2018 (LMO cell).

    Verified on 2026-11-19 against Table I of the paper (DOI
    10.1109/TSG.2016.2578950, p. 7): all ten values below match. They were
    fitted to one LMO cell; Table I's DoD model is LMO-specific, and the
    temperature model is only recommended above 15 C (Sec. III-A). Two
    differences from the paper: calendar ageing is integrated over the SoC and
    temperature trace, where the paper evaluates it at their averages (Eq. 4),
    and the paper's text says T_ref "293K or 25C" while Table I gives 25 C,
    which is used here. k_sigma and k_T are swept in pipeline/08_revision.py.

    Linearised damage f accumulates from calendar ageing,
        df_cal = k_t * S_sigma(soc) * S_T(temp) * dt,
    and from each rainflow cycle of depth d, mean soc s and mean temperature T,
        df_cyc = n * S_delta(d) * S_sigma(s) * S_T(T),
    with S_delta(d) = 1 / (k_d1 * d^k_d2 + k_d3), S_sigma(s) = exp(k_sigma (s - s_ref)),
    S_T(T) = exp(k_T (T - T_ref) T_ref / T) in kelvin. Capacity fade is
        L(f) = 1 - alpha_sei exp(-beta_sei f) - (1 - alpha_sei) exp(-f).
    """
    k_t: float = 4.14e-10
    k_sigma: float = 1.04
    sigma_ref: float = 0.5
    k_T: float = 6.93e-2
    T_ref_c: float = 25.0
    k_d1: float = 1.4e5
    k_d2: float = -0.501
    k_d3: float = -1.23e5
    alpha_sei: float = 5.75e-2
    beta_sei: float = 121.0
    eol_fade: float = 0.20
    deployment_years: float = 10.0
    battery_kgco2e_per_kwh: float = 85.0     # ASSUMPTION, LCA literature range
    # Paper revision: also score each trace under these k_sigma / k_T values
    # (env/battery.py:aging_variants). Empty = off, as in the first runs.
    variant_k_sigmas: Tuple[float, ...] = ()
    variant_k_Ts: Tuple[float, ...] = ()
    # SolCascade: also score each trace under these chemistries ("lmo", "lfp",
    # "nmc"; env/chemistry.py). Empty = off, as in all earlier runs.
    chemistries: Tuple[str, ...] = ()


@dataclass
class SolarCfg:
    # "pvgis" uses measured hourly irradiance and air temperature; "analytic" is
    # an offline fallback and must never be mixed into a run labelled as real.
    source: str = "pvgis"
    # Weather years are the replicates: each is a different real year at the
    # same site, so variation comes from real weather, not a random generator.
    years: Tuple[int, ...] = (2011, 2012, 2013, 2014, 2015)
    slot_minutes: int = 5
    panel_efficiency: float = 0.18
    mppt_efficiency: float = 0.90
    enclosure_gain_c: float = 15.0      # ASSUMPTION: battery temp rise at 1000 W/m2
    thermal_tau_hours: float = 1.0


@dataclass
class ControlCfg:
    # Conformal reserve (Adaptive Conformal Inference, Gibbs & Candes 2021):
    # the charge target is set so that the probability of not being able to
    # keep capturing through the coming night is at most alpha.
    alpha: float = 0.05
    aci_gamma: float = 0.01
    reserve_horizon_h: float = 36.0
    reserve_window_days: int = 30
    reserve_margin: float = 0.03        # fraction of capacity added to the target
    min_ceiling: float = 0.30
    # Refinement.
    deadline_h: float = 48.0            # a deferred frame not refined by then keeps its triage label
    V: float = 1.0                      # weight on classification value
    scarcity_beta: float = 20.0         # energy price growth below the reserve target
    urgency: float = 1.0                # priority boost as a frame nears its deadline
    min_gain: float = 0.005             # below this, finalise at triage
    max_batch: int = 500
    base_price: float = 0.002           # value per joule of energy above the reserve
    warmup_target_soc: float = 0.5      # reserve used before the forecaster has enough history
    batch_interval_min: int = 60        # how often to consider a refinement batch
    # Paper revision: score alternative reserve rules at these risk levels
    # alongside the one in use (env/forecast.py). Empty = off.
    shadow_alphas: Tuple[float, ...] = ()


@dataclass
class BaselineCfg:
    ee_theta_hi: float = 0.9            # per-frame energy-aware early exit
    ee_soc_hi: float = 0.8
    ee_soc_lo: float = 0.3
    lazy_theta: float = 0.9
    lazy_soc_on: float = 0.6
    lazy_soc_keep: float = 0.5
    ceiling_margin: float = 0.10


@dataclass
class ExperimentCfg:
    main_site: str = "cct_region"
    # Harvest-to-demand ratio: annual harvest divided by the annual energy the
    # always-refine-now policy would need at that location. It is the regime
    # coordinate every headline figure is plotted against.
    ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)
    ratio_sweep: Tuple[float, ...] = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    alpha_sweep: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2)
    days: int = 365
    n_workers: int = 0                  # 0 = all cores but one
    ci_level: float = 0.95
    n_bootstrap: int = 2000
    # Kill test: the proposed controller must beat the best per-frame
    # controller by at least this much to count as a result.
    kill_value_margin: float = 0.02     # absolute value-score points
    kill_life_ratio: float = 1.15       # battery life multiple at no worse value
    kill_value_tolerance: float = 0.01
    kill_life_tolerance: float = 0.05


# Sites. "cct_region" approximates where the Caltech Camera Traps cameras are
# (Southern California); the dataset does not publish exact coordinates, so
# this is region-level co-location. The rest are "climate transplants": the
# same real animal activity under a different sun, used only as a stress test.
SITES: Dict[str, Dict] = {
    "cct_region": dict(lat=33.0, lon=-116.8, utc_offset=-8, colocated=True),
    "phoenix":    dict(lat=33.45, lon=-112.07, utc_offset=-7, colocated=False),
    "nairobi":    dict(lat=-1.29, lon=36.82, utc_offset=3, colocated=False),
    "dhaka":      dict(lat=23.81, lon=90.41, utc_offset=6, colocated=False),
    "munich":     dict(lat=48.14, lon=11.58, utc_offset=1, colocated=False),
    "bergen":     dict(lat=60.39, lon=5.32, utc_offset=1, colocated=False),
    # Snapshot Serengeti grid, central Serengeti National Park, Tanzania.
    "serengeti":  dict(lat=-2.33, lon=34.83, utc_offset=3, colocated=True),
    # SolCascade climates C1-C5 (C6 is bergen). Region-level coordinates.
    "annamites":  dict(lat=16.20, lon=107.60, utc_offset=7, colocated=True),
    "meta":       dict(lat=4.15, lon=-73.63, utc_offset=-5, colocated=True),
    "white_mtns": dict(lat=44.05, lon=-71.30, utc_offset=-5, colocated=True),
    "skukuza":    dict(lat=-24.99, lon=31.59, utc_offset=2, colocated=True),
    "nkhotakota": dict(lat=-12.90, lon=34.10, utc_offset=2, colocated=True),
    "idaho":      dict(lat=45.0, lon=-115.0, utc_offset=-7, colocated=True),
    "washington": dict(lat=48.60, lon=-119.50, utc_offset=-8, colocated=True),
    "wellington": dict(lat=-41.29, lon=174.78, utc_offset=12, colocated=True),
}

FIRST_SITES = ("cct_region", "phoenix", "nairobi", "dhaka", "munich", "bergen", "serengeti")

# Per-corpus settings, copied into a Config by `apply_corpus`.
CORPORA: Dict[str, Dict] = {
    "cct20": dict(site="cct_region", aspect=1024 / 747, vehicle_classes=("car",),
                  splits=(("train", "cis_val", "cis_test"), "trans_val", "trans_test")),
    "serengeti": dict(site="serengeti", aspect=2048 / 1536, vehicle_classes=("human", "vehicle"),
                      splits=(("train",), "calib", "eval")),
    # PREREGISTRATION_v2.md held-out tests.
    "cct_new": dict(site="cct_region", aspect=2048 / 1494, vehicle_classes=("car",),
                    splits=(("train",), "calib", "eval")),
    "serengeti11": dict(site="serengeti", aspect=2048 / 1536, vehicle_classes=("human", "vehicle"),
                        splits=(("train",), "calib", "eval"), season="S11", exclude_season="S10",
                        min_eval_cameras=10),
    # New unseen corpora (PREREGISTRATION_v3.md), each under its own recent sun.
    **{f"lila_{k}": dict(site=s, aspect=4 / 3, vehicle_classes=("human", "vehicle"),
                         splits=(("train",), "calib", "eval"), years=(2019, 2020, 2021, 2022, 2023))
       for k, s in (("swg", "annamites"), ("orinoquia", "meta"), ("ammonitor", "white_mtns"),
                    ("nkhotakota", "nkhotakota"), ("wellington", "wellington"), ("idaho", "idaho"), ("lynx", "washington"),
                    ("nz", "wellington"), ("wcs", "meta"), ("safrica", "skukuza"))},
}

# Frozen trunks and their exit taps (indices into the trunk Sequential). The
# paper uses MobileNetV3-Large; ResNet-50 is the second classifier of the
# revision: about 19x the MACs, so it also tests a different energy balance.
BACKBONES: Dict[str, Tuple[int, ...]] = {
    "mobilenet_v3_large": (6, 12, 16),
    # conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4 -> taps after layer2/3/4
    "resnet50": (5, 6, 7),
}


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    vision: VisionCfg = field(default_factory=VisionCfg)
    task: TaskCfg = field(default_factory=TaskCfg)
    node: NodeCfg = field(default_factory=NodeCfg)
    battery: BatteryCfg = field(default_factory=BatteryCfg)
    aging: AgingCfg = field(default_factory=AgingCfg)
    solar: SolarCfg = field(default_factory=SolarCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    baselines: BaselineCfg = field(default_factory=BaselineCfg)
    experiment: ExperimentCfg = field(default_factory=ExperimentCfg)
    out_dir: str = "./outputs"

    @property
    def artifacts_dir(self) -> str:
        return f"{self.out_dir}/artifacts"

    @property
    def results_dir(self) -> str:
        return f"{self.out_dir}/results"

    @property
    def slot_seconds(self) -> float:
        return 60.0 * self.solar.slot_minutes

    @property
    def slots_per_day(self) -> int:
        return int(round(1440 / self.solar.slot_minutes))

    @property
    def capacity_j(self) -> float:
        return self.battery.capacity_wh * 3600.0

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def apply_corpus(cfg: "Config", name: str) -> "Config":
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}; choose from {sorted(CORPORA)}")
    c = CORPORA[name]
    cfg.data.corpus = name
    cfg.experiment.main_site = c["site"]
    cfg.vision.aspect = c["aspect"]
    cfg.task.vehicle_classes = tuple(c["vehicle_classes"])
    train, calib, ev = c["splits"]
    cfg.data.train_splits, cfg.data.calib_split, cfg.data.eval_split = tuple(train), calib, ev
    if "season" in c:
        cfg.data.serengeti_season = c["season"]
        cfg.data.serengeti_exclude_season = c.get("exclude_season", "")
    if "years" in c:
        cfg.solar.years = tuple(c["years"])
    if "min_eval_cameras" in c:
        cfg.data.serengeti_min_eval_cameras = c["min_eval_cameras"]
    return cfg


def apply_backbone(cfg: "Config", name: str) -> "Config":
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    cfg.vision.backbone = name
    cfg.vision.taps = BACKBONES[name]
    return cfg


def pvgis_range(cfg=None) -> Tuple[int, int]:
    """Years fetched from PVGIS. Always the full default set, so the download
    cache is shared between quick and full runs."""
    years = SolarCfg().years
    if cfg is not None and cfg.data.corpus.startswith("lila_"):
        years = cfg.solar.years
    return min(years), max(years)


def weather_sites(cfg) -> Dict[str, Dict]:
    """Sites whose weather a corpus needs: all for the first corpora, its own for a new one."""
    if cfg.data.corpus.startswith("lila_"):
        return {cfg.experiment.main_site: SITES[cfg.experiment.main_site]}
    return {k: v for k, v in SITES.items() if k in FIRST_SITES}


def quick(cfg: Config) -> Config:
    """Smoke-test preset: a subset of images, fewer years, a shorter season."""
    cfg.vision.epochs = 10
    cfg.solar.years = (2014, 2015)
    cfg.experiment.days = 90
    cfg.experiment.ratios = (0.5, 1.5)
    cfg.experiment.ratio_sweep = (0.5, 1.0, 2.0)
    cfg.experiment.alpha_sweep = (0.02, 0.1)
    cfg.experiment.n_bootstrap = 500
    return cfg
