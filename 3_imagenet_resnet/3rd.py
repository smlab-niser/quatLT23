import numpy as np
from data_loaders.imagenet import Train, Val
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
# from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
# from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.resnet_real import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from torch import nn
from utils.training import train

hparams = {
    "batch_size": 256,
    "num_epochs": 30,
    "model": "ResNet18",
    "dataset": "imagenet64",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "gpu": 0,
}

CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')

log = True

wandb_name = f"4-{hparams['model']}_{hparams['dataset']}_B={hparams['batch_size']}_O={hparams['optimizer']}_ll={hparams['learning_rate']}"


if   hparams["model"].lower() == "resnet18" : model =  ResNet18(4)
elif hparams["model"].lower() == "resnet34" : model =  ResNet34(4)
elif hparams["model"].lower() == "resnet50" : model =  ResNet50(4)
elif hparams["model"].lower() == "resnet101": model = ResNet101(4)
elif hparams["model"].lower() == "resnet152": model = ResNet152(4)
else: raise ValueError("Invalid model name")

# model = torch.load("saved_models/4-new-ResNet34_imagenet64_B=64_O=adam_ll=0.001_E=10.pth")

# if   hparams["model"].lower() == "resnet18" : model =  resnet18 (4)
# elif hparams["model"].lower() == "resnet34" : model =  resnet34 (4)
# elif hparams["model"].lower() == "resnet50" : model =  resnet50 (4)
# elif hparams["model"].lower() == "resnet101": model =  resnet101(4)
# elif hparams["model"].lower() == "resnet152": model =  resnet152(4)
# else: raise ValueError("Invalid model name")


model.to(GPU)
optimiser = torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"])
# optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

if log:
    import wandb
    wandb.init(project="QuatLT23", name="3rd.py", config=hparams)
    wandb.watch(model)

print("Loading data...")
# training_generator = torch.utils.data.DataLoader(Imagenet_Train(10000), shuffle=True, batch_size=hparams["batch_size"], num_workers=4)
# validation_generator = torch.utils.data.DataLoader(Imagenet_Val(1000), shuffle=True, batch_size=hparams["batch_size"], num_workers=4)
training_generator = torch.utils.data.DataLoader(Train(), shuffle=True, batch_size=hparams["batch_size"], num_workers=4)
validation_generator = torch.utils.data.DataLoader(Val(), shuffle=True, batch_size=hparams["batch_size"], num_workers=4)
# m = len(training_set)

batch_size = hparams["batch_size"]
num_epochs = hparams["num_epochs"]

print("Starting to Train")

train(model, num_epochs, training_generator, validation_generator, optimiser, loss_fn, GPU = GPU, log=True)

if log: wandb.finish()


# TODO:
# - the commented out train function here is good, 
#   try the one from 6_pruning/utils... if works 
#   very good, else replace it with this one