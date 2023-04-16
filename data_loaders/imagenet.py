import torch
import numpy as np
import random
import time

class Train(torch.utils.data.Dataset):
    def __init__(self, length = 1281167, base_dir="/mnt/data/datasets/imagenet/64x64", d4 = True):
        self.base_dir = base_dir+"/train"
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
        data = np.load(f"{self.base_dir}/{index}.npy")
        if self.d4:
            b = data[1:].reshape(3, 64, 64).transpose(1, 2, 0)
            x = np.dot(b, self.mat).transpose(2, 0, 1)
        else: x = data[1:].reshape(3, 64, 64)
        return torch.from_numpy(x).float(), torch.tensor([data[0]-1]).float()


class Val(torch.utils.data.Dataset):
    def __init__(self, length = 50000, base_dir="/mnt/data/datasets/imagenet/64x64", d4 = True):
        self.base_dir = base_dir+"/data/val"
        self.length = length
        data = np.load(f"{self.base_dir}/val_data", allow_pickle=True)
        self.x = torch.from_numpy((data["data"]/255).reshape(-1, 3, 64, 64)).float()
        self.y = torch.from_numpy(np.array(data["labels"])-1).float()
        if d4: self.x = make4D(self.x)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def make4D(x):
    """Takes an array of size (n, 3, 64, 64) and returns an array of size (n, 4, 64, 64). Adds an extra leyer to each image where the last layer is a grayscale version of the image.

    Args:
        x (np.ndarray): Array of size (n, 3, 64, 64)
        
    Returns:
        np.ndarray: Array of size (n, 4, 64, 64)
    """

    mat = torch.tensor(
        [
            [1, 0, 0, 0.299],
            [0, 1, 0, 0.587],
            [0, 0, 1, 0.144]
        ]
    ).float().to(x.device)  # RGB to RGB+Grayscale conversion matrix
    x = x.permute(0, 2, 3, 1)
    x = torch.matmul(x, mat)
    return x.permute(0, 3, 1, 2)