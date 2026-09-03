"""Elastic backbone: one set of weights serving k resolutions x m early exits.

Two mechanisms, both taken from the literature the proposal already cites:
  * resolution-aware / switchable BatchNorm  (Slimmable Nets, DRNet)
  * weighted multi-exit cross-entropy        (BranchyNet)

Only the BN statistics are private per resolution -- under 1% parameter
overhead -- so a single checkpoint serves every operating point.
"""
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResHolder:
    """Module-global current-resolution index, so BN can switch without
    threading an argument through every forward()."""
    idx = 0


def set_resolution_idx(i: int):
    _ResHolder.idx = int(i)


def get_resolution_idx() -> int:
    return _ResHolder.idx


class SwitchableBN2d(nn.Module):
    """Private BN statistics per input resolution (Yu et al., ICLR'19)."""

    def __init__(self, num_features: int, n_switches: int):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features) for _ in range(n_switches)])

    def forward(self, x):
        return self.bns[_ResHolder.idx](x)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, n_switches):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = SwitchableBN2d(cout, n_switches)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = SwitchableBN2d(cout, n_switches)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                SwitchableBN2d(cout, n_switches),
            )

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + idt, inplace=True)


class ExitHead(nn.Module):
    """Cheap classifier branch. Kept to pool+linear so the exit itself does not
    dominate the energy of exiting early."""

    def __init__(self, cin, n_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class ElasticNet(nn.Module):
    """Multi-resolution, multi-exit ResNet.

    forward(x, upto) runs only as far as exit `upto` -- this is what makes the
    per-exit energy measurement honest, since we never pay for blocks we skip.
    """

    def __init__(self, resolutions: Tuple[int, ...], n_exits: int = 3,
                 base_width: int = 16, blocks_per_stage: int = 2,
                 n_classes: int = 100):
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.n_switches = len(self.resolutions)
        self.n_exits = n_exits
        self.n_classes = n_classes

        s = self.n_switches
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, 3, 1, 1, bias=False),
            SwitchableBN2d(base_width, s),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        self.exits = nn.ModuleList()
        cin = base_width
        for st in range(n_exits):
            cout = base_width * (2 ** st)
            stride = 1 if st == 0 else 2
            blocks = [BasicBlock(cin, cout, stride, s)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(BasicBlock(cout, cout, 1, s))
            self.stages.append(nn.Sequential(*blocks))
            self.exits.append(ExitHead(cout, n_classes))
            cin = cout

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x, upto: int = None) -> List[torch.Tensor]:
        """Returns logits for exits 0..upto (inclusive). upto=None -> all."""
        upto = self.n_exits - 1 if upto is None else int(upto)
        h = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            h = stage(h)
            outs.append(self.exits[i](h))
            if i >= upto:
                break
        return outs

    def resize_for(self, x, res_idx: int) -> torch.Tensor:
        """Resample an input batch to the resolution of switch `res_idx`."""
        r = self.resolutions[res_idx]
        if x.shape[-1] == r:
            return x
        return F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False)


def multi_exit_loss(logits_list, target, weights) -> torch.Tensor:
    """BranchyNet's weighted sum of per-exit cross-entropies."""
    loss = 0.0
    for i, lg in enumerate(logits_list):
        loss = loss + float(weights[i]) * F.cross_entropy(lg, target)
    return loss


def sandwich_indices(n_res: int, n_random: int, rng) -> List[int]:
    """Universally-Slimmable sandwich rule, applied to the resolution axis:
    always train the smallest and largest switch, plus n random ones."""
    idxs = [0, n_res - 1]
    if n_random > 0 and n_res > 2:
        pool = list(range(1, n_res - 1))
        rng.shuffle(pool)
        idxs += pool[:n_random]
    return sorted(set(idxs))
