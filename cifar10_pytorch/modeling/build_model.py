from __future__ import annotations

from cifar10_pytorch.modeling.models import build_ResNet
from cifar10_pytorch.modeling.models import VGG

_META_MODELS = {"VGG": VGG, "ResNet": build_ResNet}


def build_model(cfg):
    model = _META_MODELS[cfg.MODEL.NAME](cfg)
    return model
