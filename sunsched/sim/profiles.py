"""Hardware power profiles P1-P4 (SolCascade plan, section 5.5).

Most entries are bounded engineering estimates, not measurements; the plan says
so itself. They are sweep points, not facts about a board.
"""
from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class Profile:
    name: str
    p_sleep_w: float
    e_triage_j: float
    e_wake_cold_j: float
    e_frame_j: float
    e_wake_s3_j: Optional[float] = None      # resume from suspend-to-RAM
    p_s3_w: Optional[float] = None           # power while suspended

    def break_even_s(self) -> Optional[float]:
        """Longest gap for which staying suspended beats powering off."""
        if self.e_wake_s3_j is None:
            return None
        return (self.e_wake_cold_j - self.e_wake_s3_j) / self.p_s3_w

    def restart_j(self, gap_s: float) -> float:
        """Cheapest energy to be awake again after `gap_s` of idleness."""
        if self.e_wake_s3_j is None:
            return self.e_wake_cold_j
        return min(self.e_wake_cold_j, self.e_wake_s3_j + self.p_s3_w * gap_s)

    def batch_size(self, gap_s: Optional[float] = None) -> int:
        """Frames whose refinement costs as much as one restart."""
        e = self.e_wake_cold_j if gap_s is None else self.restart_j(gap_s)
        return math.ceil(e / self.e_frame_j)


PROFILES = {p.name: p for p in (
    Profile("P1", 0.002, 0.012, 12.0, 0.51),
    Profile("P2", 0.002, 0.015, 12.0, 0.51, 1.5, 0.150),
    Profile("P3", 0.002, 0.015, 35.0, 0.85, 6.0, 0.500),
    Profile("P4", 0.002, 0.008, 0.40, 0.08, 0.15, 0.025),
)}


def apply_profile(cfg, name: str):
    """Set the node constants so that build_node_energy returns the profile's
    joules. Every break-even is far below one 5-minute slot, so at slot
    resolution a processor is never worth keeping suspended between slots and
    the cold-wake energy is the right one (see test_profiles)."""
    p, n = PROFILES[name], cfg.node
    n.p_sleep_w = p.p_sleep_w
    n.tier_a_pj_per_mac, n.tier_a_overhead_j = 0.0, p.e_triage_j
    n.tier_b_wake_j = min(p.e_wake_cold_j, p.restart_j(cfg.slot_seconds))
    n.tier_b_active_w, n.tier_b_per_frame_j = 0.0, p.e_frame_j
    return cfg
