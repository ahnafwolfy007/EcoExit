"""Avoidable wake share: what deferral can save, from capture times alone.

N_slots is the number of slots with a capture, N_stab the fewest wakes that
serve every capture within the deadline (interval stabbing: wake at the
deadline of the oldest unserved capture; optimal, PLAN_V2 Proposition 1).
"""
import numpy as np


def stabbing_wakes(slots: np.ndarray, deadline_slots: int) -> int:
    s = np.sort(np.asarray(slots))
    n = i = 0
    while i < len(s):
        n += 1
        i = int(np.searchsorted(s, s[i] + deadline_slots, side="right"))
    return n


def wake_share(slots: np.ndarray, deadline_slots: int, wake_j: float, demand_j: float) -> dict:
    n_slots, n_stab = len(np.unique(slots)), stabbing_wakes(slots, deadline_slots)
    return dict(n_capture_slots=n_slots, n_stab=n_stab,
                wake_share=(n_slots - n_stab) * wake_j / max(demand_j, 1e-9))
