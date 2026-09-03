"""Energy accounting for the elastic backbone.

Two estimators are provided on purpose, and the gap between them is itself
one of the paper's results (Contribution D):

  * `MacsOnlyProxy`   -- the thing most "efficient inference" papers use.
  * `DecomposedModel` -- MACs + activation traffic + weight traffic + static
                          power, which is what the Millar et al. survey says
                          you actually need because data movement, not
                          compute, dominates measured energy.

`profile_model` walks every (resolution, exit) operating point of a trained
ElasticNet and measures MACs / activation bytes / weight bytes / wall time
directly via forward hooks -- nothing here is asserted, it is counted.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import numpy as np
import torch
import torch.nn as nn


@dataclass
class OpPoint:
    res_idx: int
    resolution: int
    exit_idx: int
    macs: float
    act_bytes: float
    weight_bytes: float
    time_s: float


def _count_conv_linear(module: nn.Module, out: torch.Tensor,
                        macs_acc: List[float], act_acc: List[float]):
    if isinstance(module, nn.Conv2d):
        oh, ow = out.shape[-2:]
        kh, kw = module.kernel_size
        macs = module.in_channels * module.out_channels * kh * kw * oh * ow / module.groups
        macs_acc[0] += macs
    elif isinstance(module, nn.Linear):
        macs_acc[0] += module.in_features * module.out_features
    act_acc[0] += out.numel() * out.element_size()


def profile_model(model, resolutions: Tuple[int, ...], n_exits: int,
                   set_resolution_idx, device="cpu", n_timing_runs: int = 20) -> List[OpPoint]:
    """Measure MACs, activation bytes, weight bytes and wall time for every
    (resolution, exit) pair by running real forward passes with hooks attached.
    """
    model = model.to(device).eval()
    points = []

    stem_bytes = sum(p.numel() * p.element_size() for p in model.stem.parameters())
    cum = stem_bytes
    weight_bytes_by_exit = []
    for i, stage in enumerate(model.stages):
        cum += sum(p.numel() * p.element_size() for p in stage.parameters())
        cum += sum(p.numel() * p.element_size() for p in model.exits[i].parameters())
        weight_bytes_by_exit.append(cum)

    for ri, r in enumerate(resolutions):
        set_resolution_idx(ri)
        x = torch.randn(1, 3, r, r, device=device)

        with torch.no_grad():
            for exit_idx in range(n_exits):
                macs_acc, act_acc = [0.0], [0.0]
                handles = []
                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        def hook(mod, inp, out, macs_acc=macs_acc, act_acc=act_acc):
                            _count_conv_linear(mod, out, macs_acc, act_acc)
                        handles.append(m.register_forward_hook(hook))

                _ = model(x, upto=exit_idx)
                for h in handles:
                    h.remove()

                t0 = time.perf_counter()
                for _ in range(n_timing_runs):
                    _ = model(x, upto=exit_idx)
                dt = (time.perf_counter() - t0) / n_timing_runs

                points.append(OpPoint(
                    res_idx=ri, resolution=r, exit_idx=exit_idx,
                    macs=macs_acc[0], act_bytes=act_acc[0],
                    weight_bytes=float(weight_bytes_by_exit[exit_idx]),
                    time_s=dt,
                ))
    return points


class DecomposedModel:
    """E = b0 + b1*MACs + b2*ActBytes + b3*WeightBytes + P_static*T"""

    def __init__(self, cfg):
        self.cfg = cfg

    def inference_energy_j(self, op: OpPoint) -> float:
        c = self.cfg
        e = (c.b0_j
             + c.b1_pj_per_mac * 1e-12 * op.macs
             + c.b2_pj_per_act_byte * 1e-12 * op.act_bytes
             + c.b3_pj_per_weight_byte * 1e-12 * op.weight_bytes
             + c.p_static_w * op.time_s)
        return e

    def sense_energy_j(self, resolution: int) -> float:
        kpix = (resolution * resolution) / 1000.0
        return self.cfg.e_sense_j_per_kpix * kpix + self.cfg.e_pre_j_per_kpix * kpix


class MacsOnlyProxy:
    """The proxy the proposal's own motivation section calls unreliable.
    Kept here deliberately so we can measure how badly it mis-ranks configs."""

    def __init__(self, pj_per_mac: float = 4.2):
        self.pj_per_mac = pj_per_mac

    def inference_energy_j(self, op: OpPoint) -> float:
        return self.pj_per_mac * 1e-12 * op.macs

    def sense_energy_j(self, resolution: int) -> float:
        return 0.0  # a MACs-only proxy has no notion of sensing cost at all


def build_energy_table(points: List[OpPoint], model) -> Dict:
    """Returns {(res_idx, exit_idx): energy_j}."""
    table = {}
    for op in points:
        table[(op.res_idx, op.exit_idx)] = model.inference_energy_j(op)
    return table


def add_measurement_noise(table: Dict, frac: float, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    return {k: v * (1.0 + rng.normal(0, frac)) for k, v in table.items()}


def mac_rank_mismatch(decomposed_table: Dict, macs_table: Dict) -> dict:
    """Contribution D's headline number: how often the two energy models
    disagree on which of two configs is cheaper."""
    keys = list(decomposed_table.keys())
    n_pairs = 0
    n_mismatch = 0
    cross_res_mismatch = 0
    cross_res_pairs = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            n_pairs += 1
            d_order = decomposed_table[a] < decomposed_table[b]
            m_order = macs_table[a] < macs_table[b]
            mismatch = d_order != m_order
            n_mismatch += mismatch
            if a[0] != b[0]:  # different resolution
                cross_res_pairs += 1
                cross_res_mismatch += mismatch
    from scipy.stats import spearmanr
    rho, _ = spearmanr(list(decomposed_table.values()), list(macs_table.values()))
    return dict(
        n_pairs=n_pairs, n_mismatch=n_mismatch,
        mismatch_rate=n_mismatch / max(n_pairs, 1),
        cross_res_pairs=cross_res_pairs, cross_res_mismatch=cross_res_mismatch,
        cross_res_mismatch_rate=cross_res_mismatch / max(cross_res_pairs, 1),
        spearman_rho=float(rho),
    )
