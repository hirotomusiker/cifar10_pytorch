import math

import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from cifar10_pytorch.modeling.functions import swish, Drop_Connect
from cifar10_pytorch.config.paths_catalog import get_model_urls

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block.
    """
    def __init__(self, in_ch, out_ch, expansion, kernel_size, stride, drop_connect_rate=0.2):
        """
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            expansion (int): channel expansion rate
            kernel_size (int): kernel size of depthwise conv layer
            stride (int): stride of depthwise conv layer
            drop_connect_rate:
        """
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expansion = expansion
        self.id_skip = True if stride == 1 and in_ch == out_ch else False
        self.drop_connect_rate = drop_connect_rate
        self.swish = swish()

        ch = expansion * in_ch

        if expansion != 1:
            self._expand_conv = nn.Conv2d(
                in_ch, ch,
                kernel_size=1, stride=1,
                padding=0, bias=False)
            self._bn0 = nn.BatchNorm2d(ch, eps=1e-3, momentum=0.01)

        self._depthwise_conv = nn.Conv2d(ch, ch, kernel_size=kernel_size,
                               stride=stride, padding=(kernel_size-1)//2, groups=ch, bias=False)
        self._bn1 = nn.BatchNorm2d(ch, eps=1e-3, momentum=0.01)

        # SE layers
        self._se_reduce = nn.Conv2d(ch, in_ch//4, kernel_size=1)
        self._se_expand = nn.Conv2d(in_ch//4, ch, kernel_size=1)

        self._project_conv = nn.Conv2d(
            ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self._bn2 = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)

        self._drop_connect = Drop_Connect(self.drop_connect_rate)

    def forward(self, inputs):
        x = inputs
        if self.expansion != 1:
            x = self.swish(self._bn0(self._expand_conv(x)))
        h = self.swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        se = F.avg_pool2d(h, h.size(2))
        se = self.swish(self._se_reduce(se))
        se = self._se_expand(se).sigmoid()
        h = h * se

        h = self._bn2(self._project_conv(h))

        # Skip Connection
        if self.id_skip:
            if self.training and self.drop_connect_rate > 0:
                h = self._drop_connect(h)
            h = h + inputs
        return h


class EfficientNet(nn.Module):
    """
    EfficientNet model. https://arxiv.org/abs/1905.11946
    """
    def __init__(self, block_args, num_classes=10):
        super(EfficientNet, self).__init__()
        self.block_args = block_args
        self._conv_stem = nn.Conv2d(
            3,
            block_args["stem_ch"],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self._bn0 = nn.BatchNorm2d(block_args["stem_ch"], eps=1e-3, momentum=0.01)
        self._blocks = self._make_blocks()
        self._conv_head = nn.Conv2d(
            block_args["head_in_ch"],
            block_args["head_out_ch"],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self._bn1 = nn.BatchNorm2d(block_args["head_out_ch"], eps=1e-3, momentum=0.01)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self.block_args["dropout_rate"])
        self._fc = nn.Linear(block_args["head_out_ch"], num_classes)
        self.swish = swish()

    def _make_blocks(self):
        layers = []
        for n in range(7):
            strides = [self.block_args["stride"][n]] \
                     + [1] * (self.block_args["num_repeat"][n] - 1)
            in_chs = [self.block_args["input_ch"][n]] \
                     + [self.block_args["output_ch"][n]] * (self.block_args["num_repeat"][n] - 1)
            for stride, in_ch in zip(strides, in_chs):
                layers.append(
                    MBConvBlock(
                        in_ch,
                        self.block_args["output_ch"][n],
                        self.block_args["expand_ratio"][n],
                        self.block_args["kernel_size"][n],
                        stride,
                        drop_connect_rate=self.block_args["drop_connect_rate"],
                    )
                )
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.swish(self._bn0(self._conv_stem(x)))
        h = self._blocks(h)
        h = self.swish(self._bn1(self._conv_head(h)))
        h = self._avg_pooling(h)
        h = h.view(h.size(0), -1)
        h = self._dropout(h)
        h = self._fc(h)
        return h


def round_filters(ch, multiplier, divisor=8):
    """
    channel number scaling for EfficientNet.
    Args:
        ch (int): number of channel to scale
        multiplier (float): scaling factor
        divisor (int): divisor of scaled number of channels
    Returns:
        ch_scaled (int): scaled number of channels

    """
    ch *= multiplier
    ch_scaled = int(ch + divisor / 2) // divisor * divisor
    if ch_scaled < 0.9 * ch:
        ch_scaled += divisor
    return ch_scaled


def build_EfficientNet(cfg):
    """
    build EfficientNetB0-B7 from the input configuration.
    """
    block_args = {
        "num_repeat": [1, 2, 2, 3, 3, 4, 1],
        "kernel_size": [3, 3, 5, 3, 5, 5, 3],
        "stride": [1, 2, 2, 2, 1, 2, 1],
        "expand_ratio": [1, 6, 6, 6, 6, 6, 6],
        "input_ch": [32, 16, 24, 40, 80, 112, 192],
        "output_ch": [16, 24, 40, 80, 112, 192, 320],
        "dropout_rate": cfg.MODEL.DROPOUT_RATE,
        "drop_connect_rate": cfg.MODEL.DROPCONNECT_RATE,
        "stem_ch": round_filters(32, cfg.MODEL.CHANNEL_MULTIPLIER),
        "head_in_ch": round_filters(320, cfg.MODEL.CHANNEL_MULTIPLIER),
        "head_out_ch": round_filters(1280, cfg.MODEL.CHANNEL_MULTIPLIER),
    }

    # Scale number of blocks
    if cfg.MODEL.DEPTH_MULTIPLIER > 1.0:
        block_args["num_repeat"] = [
            math.ceil(n * cfg.MODEL.DEPTH_MULTIPLIER) for n in block_args["num_repeat"]
        ]

    # Scale number of channels
    if cfg.MODEL.CHANNEL_MULTIPLIER > 1.0:
        block_args["input_ch"] = [
            round_filters(n, cfg.MODEL.CHANNEL_MULTIPLIER) for n in block_args["input_ch"]
        ]

        block_args["output_ch"] = [
            round_filters(n, cfg.MODEL.CHANNEL_MULTIPLIER) for n in block_args["output_ch"]
        ]

    # Load ImageNet-pretrained model
    if cfg.MODEL.PRETRAINED:
        model_urls = get_model_urls()
        state_dict = model_zoo.load_url(model_urls[cfg.MODEL.PRETRAINED])
        model = EfficientNet(block_args, num_classes=1000)
        model.load_state_dict(state_dict)
        model._fc = nn.Linear(
            round_filters(1280, cfg.MODEL.CHANNEL_MULTIPLIER),
            10
        )
    else:
        model = EfficientNet(block_args)
    return model

