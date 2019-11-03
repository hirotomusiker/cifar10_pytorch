from cifar10_pytorch.modeling.models import *

_META_MODELS = {
    "VGG": VGG,
    "ResNet": build_ResNet,
    "EfficientNet": build_EfficientNet,
}


def build_model(cfg):
    model = _META_MODELS[cfg.MODEL.NAME](cfg)
    return model