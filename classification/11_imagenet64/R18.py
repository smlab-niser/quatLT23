import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange

from models.resnet_real import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.resnet_quat import ResNet18_quat, ResNet34_quat, ResNet50_quat, ResNet101_quat, ResNet152_quat
from utils.pruning import prune_model, reset_model
from utils.training import train_accuracy, train_multiple_models
from data_loaders.imagenet import Train, Val

class LR_Sched:
    def __init__(self, optimiser, lr_schedule):
        self.optimiser = optimiser
        self.lr_schedule = lr_schedule

    def step(self, epoch):
        if epoch in self.lr_schedule:
            self.optimiser.param_groups[0]["lr"] = self.lr_schedule[epoch]
            print(f"Learning rate changed to {self.optimiser.param_groups[0]['lr']}")

hparams = {
    "batch_size": 256,
    "num_epochs": 15,
    # "num_prune": 25,
    # "left_after_prune": 0.7,
    "model": "ResNet18_real",
    "dataset": "imagenet64x64",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "gpu": 0,
    "lr_schedule": {
        0: 0.1,
        11: 0.01,
        13: 0.001,
    }
}


log = False
save = False
seed = 21
save_path = "saved_models/RN18_real"
CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')
num_classes = 1000

models = [
    ResNet18(4, num_classes, "RN18_real").to(GPU),
]

for model in models:
    torch.manual_seed(seed)
    model.apply(reset_model)
optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True) for model in models]
loss_fns = [nn.CrossEntropyLoss() for _ in models]
schedulers = [LR_Sched(optimiser, hparams["lr_schedule"]) for optimiser in optimisers]

if log:
    import wandb
    wandb.init(project="QuatLT23", name="RN18 real img64 last trial", config=hparams)
    for model in models:
        wandb.watch(model)

print("Loading Training data...")
training_generator = torch.utils.data.DataLoader(Train(), batch_size=hparams["batch_size"], shuffle=True)
print("Loading Validation data...")
validation_generator = torch.utils.data.DataLoader(Val(), batch_size=hparams["batch_size"], shuffle=False)


num_epochs = hparams["num_epochs"]
        

# pretraining
for epoch in range(num_epochs):
    
    for sched in schedulers:
        sched.step(epoch)

    for batch_x, batch_y in tqdm(training_generator, desc=f"Pretraining epoch {epoch+1}/{num_epochs}", unit = "batch"):
        losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
        if log: wandb.log({f"loss {model.name}": losses[i] for i, model in enumerate(models)})

    if save:
        for model in models:
            torch.save(model, f"{save_path}/{model.name}_E={epoch+1}.pt")

    if log:
        accs = {
            f"test_acc {model.name}": train_accuracy(model, validation_generator, GPU) 
            for model in models
        }
        wandb.log(accs)


# # pruning and retraining
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
#             for sched in schedulers:
#                 sched.step(epoch)

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