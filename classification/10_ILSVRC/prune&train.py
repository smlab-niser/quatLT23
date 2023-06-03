import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange

from models.resnet_real import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.resnet_quat import ResNet18_quat, ResNet34_quat, ResNet50_quat, ResNet101_quat, ResNet152_quat
from utils.pruning import prune_model, reset_model
from data_loaders.ILSVRC import Train, Val
from utils.training import train_accuracy, train_multiple_models
from torch.optim.lr_scheduler import LambdaLR as LRS
import math
import torchvision


hparams = {
    "batch_size": 256,
    "num_epochs": 120,
    "num_prune": 25,
    "left_after_prune": 0.7,
    "model": "ResNet18",
    "dataset": "ILSVRC",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "gpu": 0,
}


log = True
save = True
seed = 21 
save_path = "saved_models/RN18"
CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')
num_classes = 1000

models = [
    ResNet18_quat (4, num_classes, "RN18_quat").to(GPU),
    # ResNet18_quat(4, num_classes, "RN101_quat").to(GPU),
    # ResNet152     (4, num_classes, "RN152_real").to(GPU),
    # ResNet152_quat(4, num_classes, "RN152_quat").to(GPU),
]
for model in models:
    torch.manual_seed(seed)
    model.apply(reset_model)
optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True) for model in models]
loss_fns = [nn.CrossEntropyLoss() for _ in models]


if log:
    import wandb
    wandb.init(project="QuatLT23", name=f"{models[0].name} ILSVRC trial", config=hparams)
    for model in models:
        wandb.watch(model)

training_generator = torch.utils.data.DataLoader(Train(), batch_size=hparams["batch_size"], shuffle=True,  num_workers=4)
validation_generator = torch.utils.data.DataLoader(Val(), batch_size=hparams["batch_size"], shuffle=False, num_workers=4)


num_epochs = hparams["num_epochs"]


# pretraining
# print("Pretraining")
for epoch in range(num_epochs):
    for batch_x, batch_y in tqdm(training_generator, desc=f"Pretraining epoch {epoch+1}/{num_epochs}", unit = "batch"):
        losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)

        if log:
            losses = {f"loss {model.name}": losses[i] for i, model in enumerate(models)}
            wandb.log(losses)

    if save:
        for model in models:
            torch.save(model, f"{save_path}/{model.name}_E{epoch+1}.pt")
            # print(f"Saved {model.name}_E{epoch+1}.pt")

    accs = {
        f"test_acc {model.name}": train_accuracy(model, validation_generator, GPU) 
        for model in models
    }

    if log: wandb.log(accs)
    else: print(accs)

# pruning and retraining
# for prune_it in range(hparams["num_prune"]):

#     models = [prune_model(model, 1-hparams["left_after_prune"]) for model in models]
#     for model in models:
#         torch.manual_seed(seed)
#         model.apply(reset_model)

#     optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True) for model in models]
#     loss_fns = [nn.CrossEntropyLoss() for _ in models]


#     for epoch in trange(num_epochs, desc=f"Pruning {prune_it+1}/{hparams['num_prune']}", unit = "epoch"):
#         for batch_x, batch_y in training_generator:
#             losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)

#         if log:
#             losses = {f"loss {model.name}": losses[i] for i, model in enumerate(models)}
#             accs = {
#                 f"test_acc {model.name}": train_accuracy(model, validation_generator, GPU) 
#                 for model in models
#             }
#             wandb.log(losses | accs)

#         if save:
#             for model in models:
#                 torch.save(model, f"{save_path}/{model.name}_{prune_it+1}.pt")


if log:
    wandb.finish()