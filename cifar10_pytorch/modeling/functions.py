import torch
import torch.nn as nn


class swish(nn.Module):
    """
    swish activation. https://arxiv.org/abs/1710.05941
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class hswish(nn.Module):
    """
    h-swish activation. https://arxiv.org/abs/1905.02244
    """
    def __init__(self):
        super(hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6

class Drop_Connect:
    def __init__(self, drop_connect_rate):
        self.keep_prob = 1.0 - torch.tensor(drop_connect_rate, requires_grad=False)

    def __call__(self, x):
        drop_mask = torch.rand(x.shape[0], 1, 1, 1) + self.keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
        return drop_mask * x / self.keep_prob