"""CIFAR-100 loading for the elastic backbone.

CIFAR-100 is used only as the labelled-image pool -- Hole H4 in the report is
explicit that CIFAR-100 itself has no temporal structure and should not be
read as the deployment workload. The *timing* of which frames matter comes
from `ecoexit.sim.stream` instead; this module only supplies real images and
real labels for whichever CIFAR-100 index the stream points at.

Images are kept at native 32x32 and resized on the fly to each backbone
resolution, exactly mirroring what a camera pipeline does before inference.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


def _base_transform():
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


def get_datasets(root: str = "./data", download: bool = True):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = _base_transform()
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=train_tf)
    test = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=test_tf)
    return train, test


def make_loaders(train_cfg, root: str = "./data"):
    train_ds, test_ds = get_datasets(root)

    if train_cfg.train_subset and train_cfg.train_subset < len(train_ds):
        rng = np.random.default_rng(train_cfg.seed)
        idx = rng.choice(len(train_ds), size=train_cfg.train_subset, replace=False)
        train_ds = Subset(train_ds, idx.tolist())

    n_val = int(len(train_ds) * train_cfg.val_fraction)
    n_train = len(train_ds) - n_val
    gen = torch.Generator().manual_seed(train_cfg.seed)
    train_split, val_split = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_split, batch_size=train_cfg.batch_size, shuffle=True,
                               num_workers=train_cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_split, batch_size=256, shuffle=False,
                             num_workers=train_cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=train_cfg.num_workers)
    return train_loader, val_loader, test_loader, test_ds
