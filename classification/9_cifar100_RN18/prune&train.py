import numpy as np
import torch
from torch import nn
from tqdm import trange

from models.resnet_real import ResNet18, ResNet34, ResNet50
from models.resnet_quat import ResNet18_quat, ResNet34_quat, ResNet50_quat
from utils.pruning import prune_model, reset_model
from utils.training import train_accuracy, train_multiple_models
from torch.optim.lr_scheduler import LambdaLR as LRS
import math
import torchvision
from torchvision import transforms

from oth import *

hparams = {
    "batch_size": 100,
    "num_epochs": 100,
    "warmup_epochs": 10,
    "num_prune": 20,
    "left_after_prune": 0.7,
    "model": "ResNet50",
    "dataset": "cifar100",
    "version": "fine",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "gpu": 2,
}


log = True
save = True
seed = 21
save_path = "saved_models"
CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')
num_classes = 100

models = [
    # ResNet18      (4, num_classes, "RN18_real").to(GPU),
    # ResNet18_quat (4, num_classes, "RN18_quat").to(GPU),
    # ResNet34      (4, num_classes, "RN34_real").to(GPU),
    # ResNet34_quat (4, num_classes, "RN34_quat").to(GPU),
    ResNet50      (4, num_classes, "RN50_real").to(GPU),
    ResNet50_quat (4, num_classes, "RN50_quat").to(GPU),
]

for model in models:
    torch.manual_seed(seed)
    model.apply(reset_model)
optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True) for model in models]
loss_fns = [nn.CrossEntropyLoss() for _ in models]
scheduler = [
    LRS(
        optimiser,
        lr_lambda=adjust_learning_rate
    ) for optimiser in optimisers
]

if log:
    import wandb
    wandb.init(project="QuatLT23", name="RN18,34,50 real+quat cifar100 pruning", config=hparams)
    for model in models:
        wandb.watch(model)

trainset = torchvision.datasets.CIFAR100(root='/home/aritra/project/quatLT23/data/cifar100', train=True, download=True, transform=transform_train)
training_generator = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, drop_last=True)

testset = torchvision.datasets.CIFAR100(root='/home/aritra/project/quatLT23/data/cifar100', train=False, download=True, transform=transform_test)
validation_generator = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, drop_last=True)

num_epochs = hparams["num_epochs"]

# pretraining
for epoch in trange(num_epochs, desc="Pre Training Full model"):
    for batch_x, batch_y in training_generator:
        losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
        for sched in scheduler:
            sched.step()

    # real_test_acc = train_accuracy(models[0], validation_generator, GPU)
    # quat_test_acc = train_accuracy(models[1], validation_generator, GPU)
    
    # test_accs = [train_accuracy(model, validation_generator, GPU) for model in models]

    if log:
        losses = {f"loss {model.name}": losses[i] for i, model in enumerate(models)}
        accs = {
            f"test_acc {model.name}": train_accuracy(model, validation_generator, GPU) 
            for model in models
        }
        wandb.log(losses | accs)

    if save:
        for model in models:
            torch.save(model, f"{save_path}/{model.name}_prune/{model.name}_unpruned.pt")


# pruning and retraining
for prune_it in range(hparams["num_prune"]):

    models = [prune_model(model, 1-hparams["left_after_prune"]) for model in models]
    for model in models:
        torch.manual_seed(seed)
        model.apply(reset_model)

    optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True) for model in models]
    loss_fns = [nn.CrossEntropyLoss() for _ in models]
    scheduler = [
        LRS(
            optimiser,
            lr_lambda=adjust_learning_rate
        ) for optimiser in optimisers
    ]


    for epoch in trange(num_epochs, desc=f"Pruning {prune_it+1}/{hparams['num_prune']}", unit = "epoch"):
        for batch_x, batch_y in training_generator:
            losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
            for sched in scheduler:
                sched.step()

        if log:
            losses = {f"loss {model.name}": losses[i] for i, model in enumerate(models)}
            accs = {
                f"test_acc {model.name}": train_accuracy(model, validation_generator, GPU) 
                for model in models
            }
            wandb.log(losses | accs)

        if save:
            for model in models:
                torch.save(model, f"{save_path}/{model.name}_prune/{model.name}_{prune_it+1}.pt")


if log:
    wandb.finish()