import numpy as np
import torch
from torch import nn
from tqdm import trange

from models.resnet_real import ResNet18
from models.resnet_quat import ResNet18_quat
from utils.training import train_accuracy, train_multiple_models
from torch.optim.lr_scheduler import LambdaLR as LRS
import math
import torchvision
from torchvision import transforms

hparams = {
    "batch_size": 100,
    "num_epochs": 120,
    "warmup_epochs": 10,
}

def adjust_learning_rate(epoch):
    batch_idx = 0
    data_nums = 50000//hparams["batch_size"]
    if epoch < hparams["warmup_epochs"]:
        epoch += float(batch_idx + 1) / data_nums
        lr_adj = 1. * (epoch / hparams["warmup_epochs"])
    else:
        run_epochs = epoch - hparams["warmup_epochs"]
        total_epochs = hparams["num_epochs"] - hparams["warmup_epochs"]
        T_cur = float(run_epochs * data_nums) + batch_idx
        T_total = float(total_epochs * data_nums)
        lr_adj = 0.5 * (1 + math.cos(math.pi * T_cur / T_total))
    return lr_adj

class To4Channels:
    def __init__(self):
        pass
    def __call__(self, sample):
        mat = np.array(
            [
                [1, 0, 0, 0.299],
                [0, 1, 0, 0.587],
                [0, 0, 1, 0.144]
            ]
        )
        sample = sample.numpy().transpose(1, 2, 0)
        
        # torch.Tensor(np.dot(batch_x.numpy().transpose(0, 2, 3, 1), mat).transpose(0, 3, 1, 2)).float()
        
        return torch.from_numpy(np.dot(sample, mat).transpose(2, 0, 1)).float()


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    To4Channels(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    To4Channels(),
])