from __future__ import annotations

import torch
import torchvision

from .transform import build_transforms


def prepare_cifar10_dataset(cfg):
    """prepare CIFAR10 dataset based on configuration"""

    transform_train = build_transforms(cfg, is_train=True)
    transform_test = build_transforms(cfg, is_train=False)

    dataset_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.SOLVER.BATCHSIZE,
        shuffle=True,
        num_workers=cfg.SOLVER.NUM_WORKERS,
    )
    dataset_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.BATCHSIZE,
        shuffle=False,
        num_workers=cfg.SOLVER.NUM_WORKERS,
    )

    return dataloader_train, dataloader_test
