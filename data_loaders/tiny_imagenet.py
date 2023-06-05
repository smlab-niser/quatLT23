from json import load
import torch
import numpy as np

class Train(torch.utils.data.Dataset):
    def __init__(self, length = 100000, d4 = True):
        with open("/home/aritra/project/quatLT23/base_dirs.json") as f:
            self.base_dir = load(f)["tiny_imagenet"]
        self.length = length
        if d4: self.x = torch.from_numpy(np.load(f"{self.base_dir}/x_train_4.npy"))
        else:  self.x = torch.from_numpy(np.load(f"{self.base_dir}/x_train_3.npy"))
        self.y = torch.from_numpy(np.load(f"{self.base_dir}/y_train.npy"))
    def __len__(self): return self.length
    def __getitem__(self, index): return self.x[index], self.y[index]

class Val(torch.utils.data.Dataset):
    def __init__(self, length = 10000, d4 = True):
        with open("/home/aritra/project/quatLT23/base_dirs.json") as f:
            self.base_dir = load(f)["tiny_imagenet"]
        self.length = length
        if d4: self.x = torch.from_numpy(np.load(f"{self.base_dir}/x_val_4.npy"))
        else:  self.x = torch.from_numpy(np.load(f"{self.base_dir}/x_val_3.npy"))
        self.y = torch.from_numpy(np.load(f"{self.base_dir}/y_val.npy"))
    def __len__(self): return self.length
    def __getitem__(self, index): return self.x[index], self.y[index]