"""Energy cost of every node action, from measured MACs and assumed hardware."""
from dataclasses import dataclass
from typing import Dict


@dataclass
class NodeEnergy:
    sleep_j: float                 # per slot, always paid
    capture_day_j: float
    capture_night_j: float
    triage_j: float                # tier A: cheapest exit at low resolution
    store_j: float
    wake_j: float                  # tier B wake, paid once per slot in which it runs
    refine_j: Dict[str, float]     # tier B, per frame, by refinement operating point

    def capture_j(self, night: bool) -> float:
        return self.capture_night_j if night else self.capture_day_j


def build_node_energy(node_cfg, macs: Dict[str, int], slot_seconds: float) -> NodeEnergy:
    """`macs` holds measured MAC counts for "triage" and each refine op name."""
    tier_b = {}
    for op, m in macs.items():
        if op == "triage":
            continue
        seconds = m / (node_cfg.tier_b_gmac_per_s * 1e9)
        tier_b[op] = node_cfg.tier_b_per_frame_j + node_cfg.tier_b_active_w * seconds
    return NodeEnergy(
        sleep_j=node_cfg.p_sleep_w * slot_seconds,
        capture_day_j=node_cfg.e_capture_day_j,
        capture_night_j=node_cfg.e_capture_night_j,
        triage_j=node_cfg.tier_a_overhead_j + macs["triage"] * node_cfg.tier_a_pj_per_mac * 1e-12,
        store_j=node_cfg.e_store_j,
        wake_j=node_cfg.tier_b_wake_j,
        refine_j=tier_b,
    )
