import torch
import torch.nn as nn
from htorch import layers
import torch.nn.functional as F

"""
Conv-6 architecture.
"""


def std_hparams():
    hparams = {
        "dataset": 'cifar10',
        "training": {
            "batch_size": 60,
            "num_epochs": 60,
            "learning_rate": 3e-4,
            "optimizer": "adam"
        },
        "pruning": {
            "iterations": 20,
            "percentage": 0.2
        }
    }
    return hparams


class Real(nn.Module):
    def __init__(self, out_channels: int = 10):
        super().__init__()
        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out_channels)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool(x)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.pool(x)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Quat(nn.Module):
    def __init__(self, out_channels: int = 10):
        super().__init__()
        self.conv11 = layers.QConv2d(1, 16, kernel_size=3,
                                     stride=1, padding=1)
        self.conv12 = layers.QConv2d(16, 16, kernel_size=3,
                                     stride=1, padding=1)
        self.conv21 = layers.QConv2d(16, 32, kernel_size=3,
                                     stride=1, padding=1)
        self.conv22 = layers.QConv2d(32, 32, kernel_size=3,
                                     stride=1, padding=1)
        self.conv31 = layers.QConv2d(32, 64, kernel_size=3,
                                     stride=1, padding=1)
        self.conv32 = layers.QConv2d(64, 64, kernel_size=3,
                                     stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = layers.QLinear(64 * 4 * 4, 64)
        self.fc2 = layers.QLinear(64, 64)
        self.fc3 = nn.Linear(256, out_channels)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool(x)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.pool(x)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
