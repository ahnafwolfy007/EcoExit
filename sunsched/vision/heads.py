"""Exit heads trained on cached features, plus temperature scaling."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExitHead(nn.Module):
    def __init__(self, d_in: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainedHead:
    head: ExitHead
    mean: np.ndarray
    std: np.ndarray
    temperature: float = 1.0

    def logits(self, X: np.ndarray, batch: int = 4096) -> np.ndarray:
        self.head.eval()
        Z = (X.astype(np.float32) - self.mean) / self.std
        outs = []
        with torch.no_grad():
            for i in range(0, len(Z), batch):
                outs.append(self.head(torch.from_numpy(Z[i:i + batch])).numpy())
        return np.concatenate(outs) if outs else np.zeros((0, self.head.net[-1].out_features))

    def probs(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logits(X) / self.temperature)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse square-root frequency. Softer than inverse frequency, which
    over-corrects on the rare classes in CCT20 and destabilises training."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1.0)), 0.0)
    return (w / w[counts > 0].mean()).astype(np.float32)


def train_head(X: np.ndarray, y: np.ndarray, n_classes: int, hidden: int, dropout: float,
               epochs: int, lr: float, weight_decay: float, seed: int,
               batch: int = 256, log=None) -> TrainedHead:
    """A fixed recipe with no tuning on held-out data: the only choices made
    after looking at held-out data are the temperatures, fitted on the
    calibration split, which never contributes a simulated deployment."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = X.mean(axis=0).astype(np.float32)
    std = (X.std(axis=0) + 1e-6).astype(np.float32)
    Z = torch.from_numpy(((X - mean) / std).astype(np.float32))
    T = torch.from_numpy(y.astype(np.int64))

    head = ExitHead(X.shape[1], hidden, n_classes, dropout)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = torch.from_numpy(class_weights(y, n_classes))

    for ep in range(epochs):
        head.train()
        perm = rng.permutation(len(Z))
        total = 0.0
        for i in range(0, len(perm), batch):
            idx = torch.from_numpy(perm[i:i + batch])
            loss = F.cross_entropy(head(Z[idx]), T[idx], weight=w, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        sched.step()
        if log and (ep == 0 or ep == epochs - 1 or (ep + 1) % 10 == 0):
            log(f"      epoch {ep + 1:3d}/{epochs}  loss {total / len(Z):.4f}")
    head.eval()
    return TrainedHead(head=head, mean=mean, std=std)


def fit_temperature(logits: np.ndarray, y: np.ndarray,
                    grid: Optional[np.ndarray] = None) -> float:
    """Temperature minimising negative log-likelihood, by grid search.

    A grid is used instead of gradient descent because it is deterministic and
    cannot diverge on a small calibration split.
    """
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 241)) if grid is None else grid
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = softmax(logits / t)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)
