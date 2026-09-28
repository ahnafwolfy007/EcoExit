"""The interface every policy implements, and what it is allowed to see.

A policy decides four things: whether to capture and how to process each
arriving frame, whether to finalise, refine now, or defer after triage, which
deferred frames to refine in the current slot, and the charge ceiling. It sees
only causal information: the battery, this slot's harvest and temperature,
the triage output of a frame it has captured, and its own queue. It never sees
a frame's true label or the future weather.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Obs:
    t: int = 0
    day: int = 0
    hour: float = 0.0
    night: bool = False
    energy_j: float = 0.0
    capacity_j: float = 1.0
    floor_j: float = 0.0
    soc: float = 0.0
    harvest_j: float = 0.0          # electrical harvest this slot, before charge losses
    temp_c: float = 25.0
    ceiling: float = 1.0
    # Battery energy that will be curtailed this slot unless something extra is
    # spent: the one moment when computing is genuinely free.
    curtail_hint_j: float = 0.0
    queue_len: int = 0
    oldest_age_slots: int = 0
    b_will_wake: bool = False       # tier B is being woken this slot anyway


@dataclass
class QItem:
    frame: int          # position in the location's stream
    t: int              # slot when captured
    gain: float         # expected value gain from refining it
    pred: int           # triage prediction
    conf: float         # triage confidence


@dataclass
class RunContext:
    cfg: object
    energy: object                  # sim.node.NodeEnergy
    forecaster: object              # env.forecast.ReserveForecaster
    slots_per_day: int
    slot_seconds: float
    deadline_slots: int
    sunrise: np.ndarray             # per-day slot index within the window, -1 if none
    sunset: np.ndarray
    n_slots: int


class Policy:
    name = "policy"
    refine_op = "full"

    def reset(self, ctx: RunContext):
        self.ctx = ctx

    def on_sunrise(self, obs: Obs):
        pass

    def ceiling(self, obs: Obs) -> float:
        return 1.0

    def pre_capture(self, obs: Obs, frame: int) -> str:
        """'triage' (default), 'direct' (skip triage, run tier B now), or 'drop'."""
        return "triage"

    def post_triage(self, obs: Obs, frame: int, pred: int, conf: float, gain: float,
                    p_animal: float) -> str:
        """'final', 'now', or 'defer'. `conf` is the species head's top-1
        probability, `p_animal` the triage detector's probability that the frame
        is not empty, `gain` the expected value of refining it."""
        return "final"

    def now_op(self, obs: Obs) -> str:
        return self.refine_op

    def wants_batch(self, obs: Obs) -> bool:
        return False

    def refine_batch(self, obs: Obs, items: List[QItem]) -> List[QItem]:
        return []

    # -- shared helpers --------------------------------------------------------
    def batch_cost_j(self, n: int, op: Optional[str] = None) -> float:
        e = self.ctx.energy
        return (0.0 if n == 0 else e.wake_j) + n * e.refine_j[op or self.refine_op]
