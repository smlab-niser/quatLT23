import torch
import pandas as pd
import numpy as np
from json import load

class Train(torch.utils.data.Dataset):
    def __init__(self, length = 50000, version = "coarse"):
        self.length = length
        with open("../base_dirs.json") as f:
            base_dir = load(f)["cifar100"]
        self.version = version == "coarse"
        if self.version:
            self.classes = 20
            self.file = base_dir + "/coarse_train.csv"
        else:
            self.classes = 100
            self.file = base_dir + "/fine_train.csv"

        train = pd.read_csv(self.file, header=None).values
        x = torch.from_numpy(train[:, 1:].reshape(-1, 3, 32, 32).transpose((0, 3, 2, 1))/255).float()
        self.y = torch.from_numpy(train[:, 0])

        mat = np.array(
            [
                [1, 0, 0, 0.299],
                [0, 1, 0, 0.587],
                [0, 0, 1, 0.144]
            ]
        )  # RGB to RGB+Grayscale conversion matrix

        self.x = torch.Tensor(np.dot(x, mat).transpose((0, 3, 1, 2))).float()

    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class Val(torch.utils.data.Dataset):
    def __init__(self, length = 10000, version = "coarse"):
        self.length = length
        with open("../base_dirs.json") as f:
            base_dir = load(f)["cifar100"]
        self.version = version == "coarse"
        if self.version:
            self.classes = 20
            self.file = base_dir + "/coarse_test.csv"
        else:
            self.classes = 100
            self.file = base_dir + "/fine_test.csv"

        test = pd.read_csv(self.file, header=None).values
        x_test = torch.from_numpy(test[:, 1:].reshape(-1, 3, 32, 32).transpose((0, 3, 2, 1))/255).float()
        self.y_test = torch.from_numpy(test[:, 0])

        mat = np.array(
            [
                [1, 0, 0, 0.299],
                [0, 1, 0, 0.587],
                [0, 0, 1, 0.144]
            ]
        )  # RGB to RGB+Grayscale conversion matrix

        self.x_test = torch.Tensor(np.dot(x_test, mat).transpose((0, 3, 1, 2))).float()

    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return self.x_test[index], self.y_test[index]