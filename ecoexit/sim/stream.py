"""Event-sparse frame-value stream.

Two sources, selected by `StreamCfg.source`:

  * "synthetic"   -- a Poisson-cluster, crepuscular-weighted arrival process.
                     Self-contained, no download, and the Phase 0 default.
  * "camera_trap" -- real capture timestamps parsed from camera-trap metadata
                     (`ecoexit.data.camera_traps`). This is the arrival process
                     of an actual deployment: real diel rhythm, real burst
                     structure, real multi-day gaps.

The synthetic path was the single limitation every adversarial review named, so
the real path is not optional for the paper. It is cheap: the Caltech metadata
is a 9 MB JSON download carrying 243,100 timestamped records, and no images are
needed to build an arrival process.

Whichever source is used, the *images* classified at those arrival times are
real, drawn from the evaluation pool by `image_index`.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FrameStream:
    value: np.ndarray          # per-slot "how much does this frame matter"
    is_event: np.ndarray       # bool
    image_index: np.ndarray    # which evaluation-pool image stands in for this slot
    source: str = "synthetic"
    location: Optional[str] = None


def _crepuscular_weight(elevation_deg: np.ndarray, boost: float) -> np.ndarray:
    """Activity peaks near the horizon (dawn/dusk), suppressed at night and
    midday -- the standard assumption for many camera-trap species."""
    e = elevation_deg
    twilight = np.exp(-0.5 * (e / 8.0) ** 2)
    night_floor = 0.15 + 0.85 * (e > -6.0)
    return night_floor * (1.0 + (boost - 1.0) * twilight)


def make_stream(elevation_deg: np.ndarray, n_slots: int, slot_seconds: int,
                stream_cfg, dataset_size: int, seed: int = 0) -> FrameStream:
    """Synthetic Poisson-cluster arrival process."""
    rng = np.random.default_rng(seed)
    days = n_slots * slot_seconds / 86400.0
    n_events_expected = max(1, int(stream_cfg.events_per_day * days))

    w = _crepuscular_weight(elevation_deg, stream_cfg.crepuscular_boost)
    p = w / w.sum()

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

    image_index = rng.integers(0, dataset_size, size=n_slots)
    return FrameStream(value=value, is_event=is_event, image_index=image_index,
                       source="synthetic")


def make_stream_from_events(event_slots: np.ndarray, n_slots: int, stream_cfg,
                            dataset_size: int, seed: int = 0,
                            location: str = None,
                            event_weights: np.ndarray = None) -> FrameStream:
    """Build a stream from real capture times already binned to slot indices.

    `event_slots` comes from `ecoexit.data.camera_traps.arrivals_to_slots`.
    `event_weights` optionally carries per-capture value (for example, a
    non-empty frame is worth more than an empty one -- 51.7% of Caltech frames
    are empty, so this distinction is most of the task).
    """
    rng = np.random.default_rng(seed)
    is_event = np.zeros(n_slots, dtype=bool)
    value = np.full(n_slots, stream_cfg.background_value)

    slots = np.asarray(event_slots, dtype=int)
    keep = (slots >= 0) & (slots < n_slots)
    slots = slots[keep]

    if event_weights is None:
        value[slots] = stream_cfg.event_value
        is_event[slots] = True
    else:
        w = np.asarray(event_weights, dtype=float)[: len(keep)][keep]
        np.maximum.at(value, slots, w)
        # Only non-empty captures count as events. Around half of all
        # camera-trap frames are empty; treating every capture as an event
        # would make "stay awake" trivially optimal.
        is_event[slots[w > stream_cfg.background_value]] = True

    image_index = rng.integers(0, dataset_size, size=n_slots)
    return FrameStream(value=value, is_event=is_event, image_index=image_index,
                       source="camera_trap", location=location)


def expected_value_curve(elevation_deg: np.ndarray, stream_cfg) -> np.ndarray:
    """The pre-capture value ESTIMATE v_hat_t every online controller acts on:
    the historical time-of-day expected frame value, NOT the realised truth of
    whether *this* frame is an event.

    A real deployment can learn "dawn and dusk are busy" from history; it cannot
    know an unseen frame's content before capturing it. Realised value credited
    to a policy always uses ground-truth `FrameStream.value`, never this.
    """
    w = _crepuscular_weight(elevation_deg, stream_cfg.crepuscular_boost)
    w_norm = (w - w.min()) / max(w.max() - w.min(), 1e-9)
    lo, hi = stream_cfg.background_value, stream_cfg.event_value
    p_event_like = 0.25 * w_norm
    return lo + p_event_like * (hi - lo)


def empirical_value_curve(is_event: np.ndarray, slots_per_day: int,
                          smooth_slots: int = 15) -> np.ndarray:
    """v_hat_t learned from the *history* of a real arrival process.

    Averages event rate by time of day across the trace, then smooths. This is
    the honest estimator for the camera-trap path: a deployed node can build
    this from its own logs, and unlike `expected_value_curve` it does not
    assume the crepuscular shape is known a priori.
    """
    n = len(is_event)
    by_tod = np.zeros(slots_per_day)
    counts = np.zeros(slots_per_day)
    for t in range(n):
        by_tod[t % slots_per_day] += float(is_event[t])
        counts[t % slots_per_day] += 1.0
    rate = by_tod / np.maximum(counts, 1.0)

    if smooth_slots > 1:
        kernel = np.ones(smooth_slots) / smooth_slots
        rate = np.convolve(np.r_[rate, rate, rate], kernel, mode="same")[slots_per_day:2 * slots_per_day]

    return np.tile(rate, int(np.ceil(n / slots_per_day)))[:n]


def value_weighted_recall(captured: np.ndarray, stream: FrameStream) -> dict:
    """captured[t] = the device was awake AND produced a usable inference."""
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
