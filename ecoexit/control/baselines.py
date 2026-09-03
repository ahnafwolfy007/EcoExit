"""Baseline controllers reimplemented so the comparison in the report is
against real competitors, not straw men. Every class here names the paper it
reimplements.

    StaticController        -- fixed config, no adaptation (ceiling / floor)
    ConfidenceOnlyExit      -- BranchyNet: exit on confidence, energy-agnostic
    ReactiveLUT             -- ePerceptive (SenSys'20): <resolution,exit> tuple
                                keyed only by current charge / SoC bucket
    HarvSchedQLearning      -- HarvNet/HarvSched (MobiSys'23): tabular Q-learning
                                over (SoC bucket, hour-of-day, day-of-week) --
                                i.e. energy state + an *implicit* time-of-day
                                forecast, no explicit harvest prediction
    MonotoneMDPController   -- Bullo et al. (2024): confidence- and
                                harvest-state-aware, monotone-in-SoC policy,
                                approximated here by value iteration over a
                                discretised (SoC, harvest-state) MDP
    OracleDP                -- perfect-foresight offline optimum: dynamic
                                programming over the *actual* realised trace
"""
from dataclasses import dataclass
from typing import List
import numpy as np


def soc_bucket(soc_frac: float, n_bins: int) -> int:
    return int(np.clip(soc_frac * n_bins, 0, n_bins - 1))


@dataclass
class Action:
    duty: int
    res_idx: int
    exit_idx: int


def all_actions(n_res: int, n_exit: int) -> List[Action]:
    acts = [Action(0, -1, -1)]
    for r in range(n_res):
        for e in range(n_exit):
            acts.append(Action(1, r, e))
    return acts


def action_energy(a: Action, energy_table, sense_energy, idle_j, wake_j) -> float:
    if a.duty == 0:
        return idle_j
    return idle_j + wake_j + sense_energy[a.res_idx] + energy_table[(a.res_idx, a.exit_idx)]


def action_value(a: Action, acc_grid, frame_value: float) -> float:
    if a.duty == 0:
        return 0.0
    return frame_value * float(acc_grid[a.res_idx, a.exit_idx])


class StaticController:
    def __init__(self, res_idx: int, exit_idx: int):
        self.a = Action(1, res_idx, exit_idx)

    def act(self, **kwargs) -> Action:
        return self.a


class ConfidenceOnlyExit:
    """BranchyNet-style: always wake at a fixed resolution, exit early only
    when confidence clears a fixed, energy-blind threshold."""

    def __init__(self, res_idx: int, thresholds: np.ndarray):
        self.res_idx = res_idx
        self.thresholds = thresholds

    def choose_exit(self, confidences: np.ndarray) -> int:
        for k, c in enumerate(confidences[:-1]):
            if c >= self.thresholds[k]:
                return k
        return len(confidences) - 1


class ReactiveLUT:
    """ePerceptive (SenSys 2020): a table keyed purely by the *current*
    charge/SoC bucket -> a fixed (resolution, exit) tuple. Tuned by grid
    search on the training site to maximise average value subject to a
    depletion penalty -- reproducing their offline table-construction step.
    """

    def __init__(self, n_bins: int, n_res: int, n_exit: int):
        self.n_bins = n_bins
        self.table = np.zeros((n_bins, 2), dtype=int)  # (res_idx, exit_idx) per bucket

    def fit(self, energy_table, sense_energy, acc_grid, idle_j, wake_j,
            depletion_penalty: float = 3.0):
        """Grid search: for each SoC bucket, pick the (r,k) with highest
        value that a node sitting at that bucket could sustain, penalising
        configs whose energy would run the bucket down further -- mirroring
        ePerceptive's charge-time-indexed table, which favours low-latency
        (cheap) configs when energy is scarce and high-fidelity configs when
        it is abundant.

        Energy is normalised to the same [0,1] scale as accuracy before the
        two are traded off -- comparing raw Joules against raw accuracy would
        let the (much larger-magnitude) energy term dominate the decision at
        every bucket regardless of urgency, collapsing the table to always
        pick the cheapest config and defeating the point of a SoC-indexed
        table at all.
        """
        n_res, n_exit = acc_grid.shape
        energies = np.array([[idle_j + wake_j + sense_energy[r] + energy_table[(r, k)]
                               for k in range(n_exit)] for r in range(n_res)])
        e_min, e_max = energies.min(), energies.max()
        e_span = max(e_max - e_min, 1e-9)

        for b in range(self.n_bins):
            frac = (b + 0.5) / self.n_bins
            urgency = depletion_penalty * (1.0 - frac) ** 2  # ~0 near full, large near empty
            best_score, best = -1e18, (0, 0)
            for r in range(n_res):
                for k in range(n_exit):
                    e_norm = (energies[r, k] - e_min) / e_span
                    score = acc_grid[r, k] - urgency * e_norm
                    if score > best_score:
                        best_score, best = score, (r, k)
            self.table[b] = best

    def act(self, soc_frac: float, **kwargs) -> Action:
        b = soc_bucket(soc_frac, self.n_bins)
        r, k = self.table[b]
        return Action(1, int(r), int(k))


class HarvSchedQLearning:
    """HarvNet/HarvSched (MobiSys 2023): tabular Q-learning over
    (SoC bucket, hour-of-day bucket, weekday) -> action. Time-of-day and
    weekday are an *implicit* forecast signal (the diurnal cycle), which is
    exactly what the report argues may already capture most of the value an
    explicit forecaster adds -- this baseline is what Contribution B's
    value-of-forecast experiment is measured against.
    """

    def __init__(self, n_soc_bins: int, n_hour_bins: int, n_actions: int,
                 lr: float = 0.2, gamma: float = 0.95, eps: float = 0.2):
        self.n_soc_bins = n_soc_bins
        self.n_hour_bins = n_hour_bins
        self.n_actions = n_actions
        self.Q = np.zeros((n_soc_bins, n_hour_bins, 2, n_actions))
        self.lr, self.gamma, self.eps = lr, gamma, eps

    def _state(self, soc_frac, hour_frac, is_weekend):
        s = soc_bucket(soc_frac, self.n_soc_bins)
        h = int(np.clip(hour_frac * self.n_hour_bins, 0, self.n_hour_bins - 1))
        return s, h, int(is_weekend)

    def act_index(self, soc_frac, hour_frac, is_weekend, rng, greedy=False) -> int:
        s, h, w = self._state(soc_frac, hour_frac, is_weekend)
        if (not greedy) and rng.random() < self.eps:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self.Q[s, h, w]))

    def update(self, soc_frac, hour_frac, is_weekend, a_idx, reward,
               next_soc_frac, next_hour_frac, next_is_weekend):
        s, h, w = self._state(soc_frac, hour_frac, is_weekend)
        ns, nh, nw = self._state(next_soc_frac, next_hour_frac, next_is_weekend)
        td_target = reward + self.gamma * self.Q[ns, nh, nw].max()
        self.Q[s, h, w, a_idx] += self.lr * (td_target - self.Q[s, h, w, a_idx])

    def decay_eps(self, factor=0.999, floor=0.02):
        self.eps = max(self.eps * factor, floor)


class MonotoneMDPController:
    """Simplified reimplementation of Bullo et al. (2024): value iteration
    over a discretised (SoC bucket, harvest-state bucket) MDP whose reward is
    per-instance confidence-weighted correctness. The optimal policy is
    monotone in SoC (their Theorem), which the discretised value-iteration
    solution reproduces empirically; harvest-state gives it reactive,
    *not* forward-looking, awareness of the current harvesting regime
    (their MDP conditions on the present harvest state, not a multi-step
    forecast) -- the same reactive/proactive line this project's Hole T3
    corrects the proposal's mischaracterisation of.
    """

    def __init__(self, n_soc_bins: int, n_harvest_bins: int, n_actions: int,
                 gamma: float = 0.97):
        self.n_soc_bins = n_soc_bins
        self.n_harvest_bins = n_harvest_bins
        self.n_actions = n_actions
        self.gamma = gamma
        self.V = np.zeros((n_soc_bins, n_harvest_bins))
        self.policy = np.zeros((n_soc_bins, n_harvest_bins), dtype=int)

    def solve(self, action_energy_j: np.ndarray, action_value_fn: np.ndarray,
              capacity_j: float, soc_min_j: float,
              harvest_levels_j: np.ndarray, harvest_trans: np.ndarray,
              n_iters: int = 200):
        """action_energy_j, action_value_fn: shape (n_actions,).
        harvest_levels_j: shape (n_harvest_bins,) representative harvest per slot.
        harvest_trans: (n_harvest_bins, n_harvest_bins) Markov transition matrix.
        """
        soc_edges = np.linspace(soc_min_j, capacity_j, self.n_soc_bins)
        for _ in range(n_iters):
            V_new = np.zeros_like(self.V)
            pol_new = np.zeros_like(self.policy)
            for si, soc in enumerate(soc_edges):
                for hi in range(self.n_harvest_bins):
                    best_q, best_a = -1e18, 0
                    for a in range(self.n_actions):
                        if action_energy_j[a] > soc - soc_min_j + harvest_levels_j[hi]:
                            continue  # infeasible: would deplete below floor
                        next_soc = np.clip(soc - action_energy_j[a] + harvest_levels_j[hi],
                                            soc_min_j, capacity_j)
                        si_next = int(np.argmin(np.abs(soc_edges - next_soc)))
                        ev = 0.0
                        for hj in range(self.n_harvest_bins):
                            ev += harvest_trans[hi, hj] * self.V[si_next, hj]
                        q = action_value_fn[a] + self.gamma * ev
                        if q > best_q:
                            best_q, best_a = q, a
                    V_new[si, hi] = best_q if best_q > -1e17 else 0.0
                    pol_new[si, hi] = best_a
            self.V, self.policy = V_new, pol_new

    def act_index(self, soc_frac: float, harvest_bin: int, capacity_j: float, soc_min_j: float) -> int:
        soc_j = soc_min_j + soc_frac * (capacity_j - soc_min_j)
        si = _soc_to_bin(soc_j, soc_min_j, capacity_j, self.n_soc_bins)
        return int(self.policy[si, int(harvest_bin)])


def _soc_to_bin(soc_j, soc_min_j: float, capacity_j: float, n_bins: int):
    """Direct index arithmetic for a uniform soc_edges = linspace(soc_min_j,
    capacity_j, n_bins) grid -- avoids an O(n_bins) argmin search per call,
    which matters once this runs inside a 60000+ slot simulation loop."""
    span = max(capacity_j - soc_min_j, 1e-9)
    idx = np.round((np.asarray(soc_j) - soc_min_j) / span * (n_bins - 1))
    return np.clip(idx, 0, n_bins - 1).astype(int)


class OracleDP:
    """Perfect-foresight offline optimum via backward dynamic programming
    over the *realised* harvest trace. Every other policy is reported as a
    percentage of this oracle -- the axis Hole H2 says the proposal is
    missing entirely.

    The backward recursion over time is inherently sequential, but the inner
    (soc_bin x action) sweep at each time step is fully vectorised with
    numpy -- this is what keeps a 45-day, 60-second-slot trace (~65k steps)
    solvable in seconds rather than minutes of pure-Python triple loops.
    """

    def __init__(self, n_soc_bins: int):
        self.n_soc_bins = n_soc_bins

    def solve(self, harvest_j: np.ndarray, action_energy_j: np.ndarray,
              base_value_per_action: np.ndarray, capacity_j: float, soc_min_j: float,
              eta_c: float, eta_d: float, frame_value_t: np.ndarray = None):
        """`base_value_per_action[a]` is 0 for sleep, acc_grid[r,k] for wake
        actions. `frame_value_t[t]` is the *ground-truth* per-slot frame
        value -- perfect foresight means the oracle is allowed to know this
        in advance (every other policy only sees the pre-capture estimate
        `expected_value_curve`, never the realised value)."""
        n = len(harvest_j)
        n_bins = self.n_soc_bins
        soc_edges = np.linspace(soc_min_j, capacity_j, n_bins)
        span = max(capacity_j - soc_min_j, 1e-9)
        if frame_value_t is None:
            frame_value_t = np.ones(n)

        V = np.zeros((n + 1, n_bins))
        pol = np.zeros((n, n_bins), dtype=np.int32)
        disch = action_energy_j / eta_d  # (n_actions,)

        for t in range(n - 1, -1, -1):
            gain = eta_c * harvest_j[t]
            next_soc = soc_edges[:, None] + gain - disch[None, :]           # (n_bins, n_actions)
            feasible = next_soc >= (soc_min_j - 1e-9)
            feasible[:, 0] = True                                          # sleep is always allowed
            next_soc_c = np.clip(next_soc, soc_min_j, capacity_j)
            idx = np.clip(np.round((next_soc_c - soc_min_j) / span * (n_bins - 1)),
                          0, n_bins - 1).astype(np.int64)
            v_next = V[t + 1][idx]                                         # (n_bins, n_actions)
            q = frame_value_t[t] * base_value_per_action[None, :] + v_next
            q = np.where(feasible, q, -1e18)
            best_a = np.argmax(q, axis=1)
            V[t] = q[np.arange(n_bins), best_a]
            pol[t] = best_a

        # forward rollout from the actual starting SoC
        soc0 = capacity_j * 0.6
        soc_traj = np.zeros(n + 1)
        soc_traj[0] = soc0
        actions = np.zeros(n, dtype=int)
        for t in range(n):
            si = int(_soc_to_bin(soc_traj[t], soc_min_j, capacity_j, n_bins))
            a = int(pol[t, si])
            actions[t] = a
            gain = eta_c * harvest_j[t]
            nxt = soc_traj[t] + gain - action_energy_j[a] / eta_d
            soc_traj[t + 1] = float(np.clip(nxt, 0.0, capacity_j))
        si0 = int(_soc_to_bin(soc0, soc_min_j, capacity_j, n_bins))
        return dict(value=V[0, si0], actions=actions, soc_traj=soc_traj[1:])
