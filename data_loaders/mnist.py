import torch
import pandas as pd
from json import load

class Train(torch.utils.data.Dataset):
    def __init__(self, length = 50000):
        self.length = length
        with open("../base_dirs.json") as f:
            base_dir = load(f)["mnist"]
        data = torch.tensor(pd.read_csv(base_dir+"/train.csv", header=None).values)
        self.x, self.y = (data[:, 1:]/255).float(), torch.nn.functional.one_hot(data[:, 0].long(), 10)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class Val(torch.utils.data.Dataset):
    def __init__(self, length = 10000):
        self.length = length
        with open("../base_dirs.json") as f:
            base_dir = load(f)["mnist"]
        test = torch.tensor(pd.read_csv(base_dir+"/test.csv", header=None).values)
        self.x_test, self.y_test = (test[:, 1:]/255).float(), torch.nn.functional.one_hot(test[:, 0].long(), 10)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x_test[index], self.y_test[index]