"""Multiply-accumulate counts per exit, measured from the real network.

These counts are the one energy input in the project that is measured rather
than assumed: the node model multiplies them by an ASSUMED energy per MAC
(NodeCfg), which is swept.
"""
from typing import Dict, Sequence

import torch
import torch.nn as nn


def trunk_macs_at_taps(trunk: nn.Sequential, taps: Sequence[int],
                       height: int, width: int) -> Dict[int, int]:
    """Cumulative MACs from the input up to and including each tap layer."""
    total = [0]

    def conv_hook(m: nn.Conv2d, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        total[0] += int(out.numel() // out.shape[0]) * (m.in_channels // m.groups) * k

    def linear_hook(m: nn.Linear, inp, out):
        total[0] += int(out.numel() // out.shape[0]) * m.in_features

    handles = []
    for m in trunk.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    taps = sorted(int(t) for t in taps)
    out = {}
    # Follows the trunk onto whatever device it is on. The counts come from
    # layer shapes, so they are identical either way.
    dev = next((p.device for p in trunk.parameters()), torch.device("cpu"))
    x = torch.zeros(1, 3, height, width, device=dev)
    with torch.no_grad():
        for i, layer in enumerate(trunk):
            x = layer(x)
            if i in taps:
                out[i] = total[0]
            if i == taps[-1]:
                break
    for h in handles:
        h.remove()
    return out


def head_macs(d_in: int, hidden: int, n_classes: int) -> int:
    return d_in * hidden + hidden * n_classes
