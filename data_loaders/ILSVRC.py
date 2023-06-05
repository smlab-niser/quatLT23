from json import load
import torch
import numpy as np

class Train(torch.utils.data.Dataset):
    def __init__(self, length = 1281166, d4 = True):
        with open("/home/aritra/project/quatLT23/base_dirs.json") as f:
            self.base_dir = load(f)["ILSVRC"]+"/train_npy2"
        self.length = length
        self.d4 = d4
        self.mat = np.array(
            [
                [1, 0, 0, 0.299],
                [0, 1, 0, 0.587],
                [0, 0, 1, 0.144]
            ]
        )
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        data = np.load(f"{self.base_dir}/img_{index}.npy")
        if self.d4:
            b = data[1:].reshape(224,224,3)
            x = np.dot(b, self.mat).transpose(2, 0, 1)
        else: x = data[1:].reshape(224, 224, 3).transpose(2,0,1)
        return torch.from_numpy(x).float(), torch.tensor([data[0]]).float()


class Val(torch.utils.data.Dataset):
    def __init__(self, length = 50000, base_dir="/mnt/data/datasets/ILSVRC", d4 = True):
        with open("/home/aritra/project/quatLT23/base_dirs.json") as f:
            self.base_dir = load(f)["ILSVRC"]+"/test_npy2"
        self.length = length
        self.d4 = d4
        self.mat = np.array(
            [
                [1, 0, 0, 0.299],
                [0, 1, 0, 0.587],
                [0, 0, 1, 0.144]
            ]
        )
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        data = np.load(f"{self.base_dir}/img_{index}.npy")
        if self.d4:
            b = data[1:].reshape(224,224,3)
            x = np.dot(b, self.mat).transpose(2, 0, 1)
        else: x = data[1:].reshape(224, 224, 3).transpose(2,0,1)
        return torch.from_numpy(x).float(), torch.tensor([data[0]]).float()