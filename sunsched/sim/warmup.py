"""Run the last days of the year before day 1, unscored.

A run that starts at 80 % charge gives a 100-day battery 80 days of energy for
free. Prepending W days of the same year removes that gift only if W is long
enough for the battery to forget its start, so W is an argument and the charge
at the start and the end of the scored year are reported.
"""
import dataclasses

import numpy as np


def extend(harvest, temp, night, elev, frame_idx, slots, w_slots):
    n = len(harvest)
    if w_slots > n:
        raise ValueError("warm-up longer than the window")
    tail = slice(n - w_slots, n)
    pre = slots >= n - w_slots
    cat = lambda a: np.concatenate([a[tail], a])
    return dict(harvest=cat(harvest), temp=cat(temp), night=cat(night), elev=cat(elev),
                frame_idx=np.concatenate([frame_idx[pre], frame_idx]),
                slots=np.concatenate([slots[pre] - (n - w_slots), slots + w_slots]),
                n_warm_frames=int(pre.sum()))


def scored_part(res, w_slots, n_warm_frames):
    k = n_warm_frames
    return dataclasses.replace(
        res, frames=res.frames[k:], final_label=res.final_label[k:], route=res.route[k:],
        capture_slot=res.capture_slot[k:] - w_slots, final_slot=np.maximum(res.final_slot[k:] - w_slots, -1),
        soc=res.soc[w_slots:], temp_c=res.temp_c[w_slots:], ceiling=res.ceiling[w_slots:])
