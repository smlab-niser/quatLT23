import torch
import torch.nn as nn
from htorch import layers
import torch.nn.functional as F

"""
LeNet_300_100 architecture.
"""

def std_hparams():
    return {
        "dataset": 'mnist',
        "training": {
            "batch_size": 60,
            "num_epochs": 80,
            "learning_rate": 1.2e-3,
            "optimizer": "adam"
        },
        "pruning": {
            "iterations": 20,
            "percentage": 0.2
        }
    }

class Real(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Quat(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = layers.QLinear(196, 75)
        self.fc2 = layers.QLinear(75, 25)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
