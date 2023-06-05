import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange

from utils.pruning import prune_model, reset_model
from utils.training import train_accuracy, train_multiple_models
from data_loaders.ILSVRC import Train, Val
from yolo_model import Yolov1, PretrainingModel

save_path = f"saved_models"

hparams = {
    "batch_size": 256,
    "num_epochs": 15,
    "num_prune": 25,
    "left_after_prune": 0.7,
    "model": "YOLO",
    "dataset": "imagenet64x64",
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "gpu": 0,
}


log = False
save = False
seed = 21
CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')


models = [
    PretrainingModel(in_channels=4, S=7).to(GPU),
]

for model in models:
    torch.manual_seed(seed)
    model.apply(reset_model)
optimisers = [torch.optim.SGD(model.parameters(), lr=0.1, momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True) for model in models]
loss_fns = [nn.CrossEntropyLoss() for _ in models]
# schedulers = [LR_Sched(optimiser, hparams["lr_schedule"]) for optimiser in optimisers]

if log:
    import wandb
    name = f"YOLO trial"
    wandb.init(project="QuatLT23", name=name, config=hparams)
    for model in models:
        wandb.watch(model)

print("Loading Training data...")
training_generator = torch.utils.data.DataLoader(Train(), batch_size=hparams["batch_size"], shuffle=True)
print("Loading Validation data...")
validation_generator = torch.utils.data.DataLoader(Val(), batch_size=hparams["batch_size"], shuffle=False)


num_epochs = hparams["num_epochs"]


# pretraining
for epoch in range(num_epochs):
    
    # for sched in schedulers:
    #     sched.step(epoch)

    for batch_x, batch_y in tqdm(training_generator, desc = f"Training {epoch+1}/{num_epochs}", unit = "batch"):
        losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
        if log: wandb.log({f"loss {model.name}": losses[i] for i, model in enumerate(models)})

    if save:
        for model in models:
            torch.save(model, f"{save_path}/{model.name}_unpruned.pt")

    if log:
        accs = {
            f"test_acc {model.name}": train_accuracy(model, validation_generator, GPU) 
            for model in models
        }
        wandb.log(accs)


# # pruning and retraining
# for prune_it in range(resume_from, hparams["num_prune"]):

#     models = [prune_model(model, 1-hparams["left_after_prune"]) for model in models]
#     for model in models:
#         torch.manual_seed(seed)
#         model.apply(reset_model)

#     optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["lr_schedule"][0], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True) for model in models]
#     loss_fns = [nn.CrossEntropyLoss() for _ in models]
#     schedulers = [LR_Sched(optimiser, hparams["lr_schedule"]) for optimiser in optimisers]


#     for epoch in trange(num_epochs, desc=f"Pruning {prune_it+1}/{hparams['num_prune']}", unit = "epoch"):

#         for sched in schedulers:
#             sched.step(epoch)

#         for batch_x, batch_y in training_generator:
#             losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
#             if log: wandb.log({f"loss {model.name}": losses[i] for i, model in enumerate(models)})

#         if save:
#             for model in models:
#                 torch.save(model, f"{save_path}/{model.name}_{prune_it+1}.pt")

#         # if log:
#         #     accs = {
#         #         f"test_acc {model.name}": train_accuracy(model, validation_generator, GPU) 
#         #         for model in models
#         #     }
#         #     wandb.log(accs)


if log:
    wandb.finish()