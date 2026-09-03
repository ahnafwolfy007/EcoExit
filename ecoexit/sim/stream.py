"""Event-sparse frame-value stream, synthesised on top of the solar clock.

This exists to answer Hole H4: CIFAR-100 has no temporal structure, no rare
events, and no notion of "the device was asleep when something happened".
A real camera-trap deployment mostly sees empty frames punctuated by bursty,
crepuscular (dawn/dusk-peaked) animal activity. We cannot ship a real
multi-year camera-trap-plus-solar-plus-battery co-located dataset in a course
project, so we synthesise the *event timing process* here (Poisson-cluster,
crepuscular-weighted) and pair each synthetic event with a real CIFAR-100
image via `assign_labels`, so classification accuracy is still measured on
real images -- only the "when do interesting things happen" timing is
synthetic. This limitation is stated in the report, not hidden.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class FrameStream:
    value: np.ndarray          # per-slot "how much does this frame matter"
    is_event: np.ndarray       # bool
    cifar_index: np.ndarray    # which CIFAR-100 test/train image stands in for this slot


def _crepuscular_weight(elevation_deg: np.ndarray, boost: float) -> np.ndarray:
    """Activity peaks when the sun is near the horizon (dawn/dusk), the
    standard assumption for many camera-trap species, and is suppressed at
    night (elevation << 0) and midday (elevation >> 0)."""
    e = elevation_deg
    twilight = np.exp(-0.5 * (e / 8.0) ** 2)          # peak near elevation 0
    night_floor = 0.15 + 0.85 * (e > -6.0)             # a little activity even at night
    return night_floor * (1.0 + (boost - 1.0) * twilight)


def make_stream(elevation_deg: np.ndarray, n_slots: int, slot_seconds: int,
                 stream_cfg, dataset_size: int, seed: int = 0) -> FrameStream:
    rng = np.random.default_rng(seed)
    days = n_slots * slot_seconds / 86400.0
    n_events_expected = max(1, int(stream_cfg.events_per_day * days))

    w = _crepuscular_weight(elevation_deg, stream_cfg.crepuscular_boost)
    p = w / w.sum()

    # Poisson-cluster process: draw burst centers, then scatter a
    # geometric-sized burst of frames around each center.
    n_bursts = max(1, int(n_events_expected / stream_cfg.burst_size_mean))
    centers = rng.choice(n_slots, size=n_bursts, replace=True, p=p)

    is_event = np.zeros(n_slots, dtype=bool)
    for c in centers:
        burst_len = max(1, rng.geometric(1.0 / stream_cfg.burst_size_mean))
        spread = rng.integers(0, 4)
        idxs = np.clip(c + rng.integers(-spread, spread + 1, size=burst_len), 0, n_slots - 1)
        is_event[idxs] = True

    value = np.full(n_slots, stream_cfg.background_value)
    value[is_event] = stream_cfg.event_value

    cifar_index = rng.integers(0, dataset_size, size=n_slots)
    return FrameStream(value=value, is_event=is_event, cifar_index=cifar_index)


def expected_value_curve(elevation_deg: np.ndarray, stream_cfg) -> np.ndarray:
    """The pre-capture value ESTIMATE v_hat_t used by every online controller
    to decide whether a slot is worth waking for -- the historical,
    time-of-day expected frame value (crepuscular weighting), NOT the
    realised ground truth of whether *this* frame is an event. A real
    deployment can learn "dawn/dusk is busy" from history; it cannot know a
    specific unseen frame's content before capturing and running inference.
    The realised outcome credited to any policy always uses the ground-truth
    `FrameStream.value`, never this estimate -- this function only shapes
    what a controller is allowed to act on *before* the fact.
    """
    w = _crepuscular_weight(elevation_deg, stream_cfg.crepuscular_boost)
    w_norm = (w - w.min()) / max(w.max() - w.min(), 1e-9)
    lo, hi = stream_cfg.background_value, stream_cfg.event_value
    # a modest, not extreme, expected-value swing: most probability mass at
    # any instant is still "probably background", scaled by relative activity
    p_event_like = 0.25 * w_norm
    return lo + p_event_like * (hi - lo)


def value_weighted_recall(captured: np.ndarray, stream: FrameStream) -> dict:
    """captured[t] = True if the device was awake AND produced a usable
    (i.e. not dropped-for-energy) inference at slot t."""
    total_value = stream.value.sum()
    captured_value = stream.value[captured].sum()
    event_total = stream.is_event.sum()
    event_captured = int((captured & stream.is_event).sum())
    return dict(
        value_recall=float(captured_value / max(total_value, 1e-9)),
        event_recall=float(event_captured / max(event_total, 1)),
        n_events=int(event_total),
        n_events_captured=event_captured,
    )
