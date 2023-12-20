"""VGG in PyTorch"""
import torch.nn as nn

_LAYER_DEFINITION = {16: [2, 2, 3, 3, 3], 19: [2, 2, 4, 4, 4]}


class VGG(nn.Module):
    def __init__(self, cfg):
        super(VGG, self).__init__()
        self.channels = [64, 128, 256, 512, 512]
        self.features = self._make_layers(_LAYER_DEFINITION[cfg.MODEL.VGG_NUM])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, definition):
        layers = []
        in_channels = 3
        for num_layers, ch in zip(definition, self.channels):
            for i in range(num_layers):
                layers += [
                    nn.Conv2d(in_channels, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                ]
                in_channels = ch
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
