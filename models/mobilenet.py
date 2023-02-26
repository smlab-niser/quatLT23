import torch
import torch.nn as nn
from htorch import layers
import torch.nn.functional as F

"""
Mobilenet V1
"""


def std_hparams():
    hparams = {
        "dataset": 'cifar10',
        "training": {
            "batch_size": 64,
            "num_epochs": 90,
            "learning_rate": 1e-2,
            "optimizer": "sgd",
            "momentum": 0.9,
            "weight_decay": 1e-4
        },
        "pruning": {
            "iterations": 20,
            "percentage": 0.2
        }
    }
    return hparams


class DW_Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride, 1
                               groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class Real(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, 2, 1),
        self.bn = nn.BatchNorm2d(32),

        self.model = nn.Sequential(
            DW_Conv(32, 64 1),
            DW_Conv(64, 128, 2),
            DW_Conv(128, 128, 1),
            DW_Conv(128, 256, 2),
            DW_Conv(256, 256, 1),
            DW_Conv(256, 512, 2),
            DW_Conv(512, 512, 1),
            DW_Conv(512, 512, 1),
            DW_Conv(512, 512, 1),
            DW_Conv(512, 512, 1),
            DW_Conv(512, 512, 1),
            DW_Conv(512, 1024, 2),
            DW_Conv(1024, 1024, 2),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = fc(x)
        return x

