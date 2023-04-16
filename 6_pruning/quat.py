import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
import numpy as np
from utils import train
from data_loaders.imagenet import Train, Val
from models.resnet_real import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.resnet_quat import ResNet18_quat, ResNet34_quat, ResNet50_quat, ResNet101_quat, ResNet152_quat


hparams = {
    "batch_size": 256,
    "num_epochs": 20,
    "model": "ResNet18_quat",
    "dataset": "imagenet64",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "gpu": 0,
}

CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')

log = True

save_name = f"4-{hparams['model']}_{hparams['dataset']}_B={hparams['batch_size']}_O={hparams['optimizer']}_ll={hparams['learning_rate']}"
# save_name = f"trial34_full"

model_name = hparams["model"]
if model_name == "ResNet18": model = ResNet18(4)
elif model_name == "ResNet34": model = ResNet34(4)
elif model_name == "ResNet50": model = ResNet50(4)
elif model_name == "ResNet101": model = ResNet101(4)
elif model_name == "ResNet152": model = ResNet152(4)
elif model_name == "ResNet18_quat": model = ResNet18_quat(4)
elif model_name == "ResNet34_quat": model = ResNet34_quat(4)
elif model_name == "ResNet50_quat": model = ResNet50_quat(4)
elif model_name == "ResNet101_quat": model = ResNet101_quat(4)
elif model_name == "ResNet152_quat": model = ResNet152_quat(4)
else: raise ValueError("Invalid model name")

# model = torch.load("saved_models/4-new-ResNet34_imagenet64_B=64_O=adam_ll=0.001_E=10.pth")

model.to(GPU)
optimiser = torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"])
# optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

if log:
    import wandb
    wandb.init(project="QuatLT23", name=save_name, config=hparams)
    wandb.watch(model)

print("Loading data...")
training_generator = torch.utils.data.DataLoader(Train(), shuffle=False, batch_size=hparams["batch_size"], num_workers=4)
validation_generator = torch.utils.data.DataLoader(Val(), shuffle=False, batch_size=hparams["batch_size"], num_workers=4)
# m = len(training_set)

batch_size = hparams["batch_size"]
num_epochs = hparams["num_epochs"]


train(model, num_epochs, training_generator, validation_generator, optimiser, loss_fn, save = save_name, GPU=GPU, log=log)

if log: wandb.finish()


