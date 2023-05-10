import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
import numpy as np
from data_loaders.ILSVRC import Train, Val
from models.resnet_real import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils.training import train

hparams = {
    "batch_size": 256,
    "num_epochs": 30,
    "model": "ResNet34",
    "dataset": "imagenet64",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "gpu": 0,
}

CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')

log = False

# save_name = f"4-new-{hparams['model']}_{hparams['dataset']}_B={hparams['batch_size']}_O={hparams['optimizer']}_ll={hparams['learning_rate']}"
save_name = f"trial34_full"

model =  ResNet34(4)
# model = torch.load("saved_models/4-new-ResNet34_imagenet64_B=64_O=adam_ll=0.001_E=10.pth")

model.to(GPU)
# optimiser = torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"])
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

if log:
    import wandb
    wandb.init(project="QuatLT23", name=save_name, config=hparams)
    wandb.watch(model)

print("Loading data...")
training_generator = torch.utils.data.DataLoader(Train(), shuffle=False, batch_size=hparams["batch_size"], num_workers=4)
validation_generator = torch.utils.data.DataLoader(Val(), shuffle=False, batch_size=hparams["batch_size"], num_workers=4)


num_epochs = hparams["num_epochs"]

train(model, num_epochs, training_generator, validation_generator, optimiser, loss_fn, log=True)

wandb.finish()
