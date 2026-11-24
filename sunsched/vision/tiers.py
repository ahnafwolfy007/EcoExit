"""Two-tier trunks for SolCascade: load, count, fine-tune, cache features.

Trunks come from timm (ImageNet weights) and end at their last feature map; the
ImageNet head is dropped. Pooled features are the average and the maximum over
the map. BatchNorm statistics stay at their ImageNet values, as in gate 1.
"""
import io
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sunsched.vision.backbone import IMAGENET_MEAN, IMAGENET_STD
from sunsched.vision.heads import class_weights

TIERS = {  # name -> (timm model, input height)
    "a_mnv4s": ("mobilenetv4_conv_small.e2400_r224_in1k", 160),
    "b_mnv3l": ("mobilenetv3_large_100.ra_in1k", 320),
    "b_mnv4m": ("mobilenetv4_conv_medium.e500_r256_in1k", 256),
    "b_mnv4h": ("mobilenetv4_hybrid_medium.ix_e550_r256_in1k", 256),
}


class Trunk(nn.Module):
    def __init__(self, timm_name: str, pretrained: bool = True):
        super().__init__()
        import timm
        self.net = timm.create_model(timm_name, pretrained=pretrained)
        for h in ("conv_head", "norm_head", "classifier", "global_pool"):     # ImageNet head, unused here
            if hasattr(self.net, h):
                setattr(self.net, h, nn.Identity())

    def forward(self, x):
        z = self.net.forward_features(x)
        return torch.cat([z.mean((2, 3)), z.amax((2, 3))], dim=1)


def count_macs(module: nn.Module, h: int, w: int) -> int:
    """Multiply-accumulates of one image (convolutions, linear layers, attention)."""
    from torch.utils.flop_counter import FlopCounterMode
    dev = next(module.parameters()).device
    module.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        module(torch.zeros(1, 3, h, w, device=dev))
    return int(fc.get_total_flops() // 2)          # the counter reports two operations per MAC


class Jpegs(Dataset):
    def __init__(self, blobs, rows, size, train):
        self.b, self.r, self.size, self.train = blobs, rows, size, train

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        im = Image.open(io.BytesIO(self.b[self.r[i]])).convert("RGB")
        h, w = self.size
        if self.train:
            s = float(torch.empty(1).uniform_(0.75, 1.0)) ** 0.5
            cw, ch = int(im.width * s), int(im.height * s)
            x0 = int(torch.randint(0, im.width - cw + 1, (1,)))
            y0 = int(torch.randint(0, im.height - ch + 1, (1,)))
            im = im.resize((w, h), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
        elif im.size != (w, h):
            im = im.resize((w, h), Image.BILINEAR)
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1)
        if self.train:
            if torch.rand(1) < 0.5:
                x = x.flip(2)
            if torch.rand(1) < 0.2:
                x = x.float().mean(0, keepdim=True).expand(3, -1, -1).to(torch.uint8)
        return x, int(self.r[i])


def norm(x, dev):
    x = x.to(dev, non_blocking=True).float() / 255.0
    return (x - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)


def finetune(trunk, size, blobs, rows, y, nc, dev, epochs=10, batch=64, lr=1.5e-4, head_lr=1e-3, seed=0,
             teacher=None, kd=0.5, T=2.0, workers=4, log=print):
    """`teacher`: logits per row index (n, nc) from tier B; adds a distillation term."""
    torch.manual_seed(seed)
    trunk.to(dev)
    with torch.no_grad():
        trunk.eval()
        d = trunk(torch.zeros(1, 3, *size, device=dev)).shape[1]
    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, nc)).to(dev)
    opt = torch.optim.AdamW([dict(params=trunk.parameters(), lr=lr), dict(params=head.parameters(), lr=head_lr)],
                            weight_decay=1e-2)
    dl = DataLoader(Jpegs(blobs, rows, size, True), batch_size=batch, shuffle=True, num_workers=workers,
                    drop_last=True, persistent_workers=workers > 0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [lr, head_lr], total_steps=max(epochs * len(dl), 2), pct_start=0.1)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    w = torch.from_numpy(class_weights(y[rows], nc)).to(dev)
    Y = torch.from_numpy(y).to(dev)
    tz = None if teacher is None else torch.from_numpy(teacher.astype(np.float32)).to(dev)
    for ep in range(epochs):
        trunk.train()
        head.train()
        for m in trunk.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        tot, seen, t0 = 0.0, 0, time.time()
        for xb, r in dl:
            r = r.to(dev)
            with torch.autocast("cuda", enabled=amp):
                z = head(trunk(norm(xb, dev)))
                loss = F.cross_entropy(z, Y[r], weight=w, label_smoothing=0.05)
                if tz is not None:
                    loss = (1 - kd) * loss + kd * T * T * F.kl_div(F.log_softmax(z / T, 1), F.softmax(tz[r] / T, 1),
                                                                   reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(trunk.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, seen = tot + float(loss.detach()) * len(r), seen + len(r)
        log(f"    epoch {ep + 1}/{epochs}  loss {tot / max(seen, 1):.3f}  {time.time() - t0:.0f} s")
    return trunk.eval(), head.eval()


@torch.no_grad()
def extract(trunk, head, size, blobs, rows, dev, batch=128, workers=4):
    dl = DataLoader(Jpegs(blobs, rows, size, False), batch_size=batch, num_workers=workers)
    X, Z = [], []
    trunk.eval()
    for xb, _ in dl:
        f = trunk(norm(xb, dev))
        X.append(f.cpu().numpy().astype(np.float16))
        Z.append(head(f).cpu().numpy().astype(np.float32))
    return np.concatenate(X), np.concatenate(Z)
