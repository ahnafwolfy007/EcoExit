"""Frozen pretrained trunk with early-exit taps.

The trunk is torchvision's MobileNetV3-Large with ImageNet weights, never
fine-tuned. Each exit is a small head on pooled features from one tap, so all
training happens on cached feature vectors and runs on a CPU.
"""
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def input_size(height: int, aspect: float) -> Tuple[int, int]:
    """(height, width), width rounded to a multiple of 32 so every stride-32
    stage divides evenly."""
    width = int(round(height * aspect / 32.0)) * 32
    return height, max(width, 32)


def load_trunk(name: str = "mobilenet_v3_large") -> nn.Sequential:
    if name != "mobilenet_v3_large":
        raise ValueError(f"unsupported backbone {name!r}")
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    trunk = model.features.eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk


def pool(x: torch.Tensor, grid: bool) -> torch.Tensor:
    """Global average and max pooling, plus a 2x2 average grid for early taps.

    The grid keeps coarse spatial layout, which matters for early exits: a
    small animal in one corner is invisible to a global average of low-level
    features but not to a quadrant average.
    """
    parts = [x.mean(dim=(2, 3)), x.amax(dim=(2, 3))]
    if grid:
        parts.append(F.adaptive_avg_pool2d(x, (2, 2)).flatten(1))
    return torch.cat(parts, dim=1)


class TapExtractor(nn.Module):
    def __init__(self, trunk: nn.Sequential, taps: Sequence[int]):
        super().__init__()
        self.trunk = trunk
        self.taps = sorted(int(t) for t in taps)
        self.last = self.taps[-1]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for i, layer in enumerate(self.trunk):
            x = layer(x)
            if i in self.taps:
                outs.append(pool(x, grid=(i != self.last)))
            if i == self.last:
                break
        return outs


def to_batch(pil_images, device=None) -> torch.Tensor:
    arr = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in pil_images])
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    if device is not None:
        t = t.to(device, non_blocking=True)
    return (t - IMAGENET_MEAN.to(t.device)) / IMAGENET_STD.to(t.device)


def tap_dims(extractor: TapExtractor, height: int, width: int) -> List[int]:
    dev = next((p.device for p in extractor.parameters()), torch.device("cpu"))
    outs = extractor(torch.zeros(1, 3, height, width, device=dev))
    return [int(o.shape[1]) for o in outs]
