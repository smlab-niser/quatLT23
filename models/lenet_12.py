import torch
import torch.nn as nn
from htorch import layers
import torch.nn.functional as F

"""
LeNet_12 architecture.
"""


def std_hparams():
    hparams = {
        "dataset": 'mnist',
        "training": {
            "batch_size": 60,
            "num_epochs": 20,
            "learning_rate": 1.2e-3,
            "optimizer": "adam"
        },
        "pruning": {
            "iterations": 20,
            "percentage": 0.2
        }
    }
    return hparams


class Real(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 12)
        self.fc2 = nn.Linear(12, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Quat(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = layers.QLinear(196, 3)
        self.fc2 = nn.Linear(12, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
