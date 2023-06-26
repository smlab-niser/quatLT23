import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange

from utils.pruning import prune_model, reset_model
from utils.training import train_accuracies, train_multiple_models
from data_loaders.ILSVRC import Train, Val
# from yolo_model import PretrainingModel as Real
# from yolo_q_model import PretrainingModel as Quat
from models.yolo_real import Pretraining as Real
from models.yolo_quat import Pretraining as Quat

save_path = f"saved_models/my_pretrain"

hparams = {
    "batch_size": 256,
    "num_epochs": 18,
    "model": "YOLO",
    "dataset": "im64",
    "optimizer": "sdg",
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.001,
    "gpu": 1,
}

# with open(f"{save_path}/dirs.sh", "a") as f:
#     f.write(f"{save_dir} ")


log = True
save = True
seed = 21
CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')


print("Initialising Models")
models = [
    Real(in_channels=4, S=7, name="real").to(GPU),
    Quat(in_channels=4, S=7, name="quat").to(GPU),
]
# raise ValueError("Don't run this file.")

for model in models:
    torch.manual_seed(seed)
    model.apply(reset_model)
optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True) for model in models]
loss_fns = [nn.CrossEntropyLoss() for _ in models]

if log:
    import wandb
    name = f"YOLO pret my impl"
    wandb.init(project="QuatLT23", name=name, config=hparams)
    for model in models:
        wandb.watch(model)

print("Loading Training Data")
training_generator = torch.utils.data.DataLoader(Train(), batch_size=hparams["batch_size"], shuffle=True, num_workers=16, pin_memory=True)
print("Loading Validation Data")
validation_generator = torch.utils.data.DataLoader(Val(), batch_size=hparams["batch_size"], shuffle=True, num_workers=16, pin_memory=True)


num_epochs = hparams["num_epochs"]


# pretraining
for epoch in range(num_epochs):
    if epoch+1 in [13, 16]:
        for optimiser in optimisers:
            optimiser.param_groups[0]["lr"] *= 0.1
            print(f"reducing lr to {optimiser.param_groups[0]['lr']}")

    for batch_x, batch_y in tqdm(training_generator, desc = f"Epoch {epoch+1}/{num_epochs}", unit = "batch"):
        losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
        if log: wandb.log({f"loss Yolo": losses[i] for i, model in enumerate(models)})

    if save:
        for model in models:
            torch.save(model, f"{save_path}_{model.name}/{epoch+1}.pt")

    if log: wandb.log(train_accuracies(models[0], validation_generator, GPU, name=models[0].name) | train_accuracies(models[1], validation_generator, GPU, name=models[1].name))

if log:
    wandb.finish()