import torch
import torch.nn as nn
from htorch import layers
import torch.nn.functional as F

"""
Conv-2 architecture.
"""


def std_hparams():
    hparams = {
        "dataset": 'cifar10',
        "training": {
            "batch_size": 60,
            "num_epochs": 40,
            "learning_rate": 2e-4,
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Quat(nn.Module):
    def __init__(self, out_channels: int = 10):
        super().__init__()
        self.conv1 = layers.QConv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = layers.QConv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = layers.QLinear(16 * 16 * 16, 64)
        self.fc2 = layers.QLinear(64, 64)
        self.fc3 = nn.Linear(256, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
