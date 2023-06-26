import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange

from utils.pruning import prune_model, reset_model
from utils.training import train_accuracies, train_multiple_models
from data_loaders.ILSVRC import Train, Val
from yolo_model import Yolov1, PretrainingModel

save_path = f"saved_models"

import argparse

parser = argparse.ArgumentParser(description="Pretraining YOLOv1 on ILSVRC.")
parser.add_argument("--gpu", type=int, choices=[0, 1, 2, 3], help="Which GPU to use.")
parser.add_argument("--lr", type=float, help="Learning rate.", default=0.01)
parser.add_argument("--momentum", type=float, help="Momentum.", default=0.9)
parser.add_argument("--weight_decay", type=float, help="Weight decay.", default=0.001)
parser.add_argument("--optimiser", type=str, help="Which optimiser to use.", default="sgd", choices=["sgd", "adam"])
parser.add_argument("--num_epochs", type=int, help="How many epochs to run.", default=20)

args = parser.parse_args()

# tmux new-session -d -s yolo1 "python pretraining.py --gpu 1 --lr 0.0001 --momentum 0.9 --weight_decay 0 --optimiser adam --num_epochs 40"
# tmux new-session -d -s yolo2 "python pretraining.py --gpu 0 --lr 0.0001 --momentum 0.9 --weight_decay 0.0001 --optimiser adam --num_epochs 40"
# tmux new-session -d -s yolo3 "python pretraining.py --gpu 0 --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --optimiser sgd --num_epochs 40"
# tmux new-session -d -s yolo4 "python pretraining.py --gpu 1 --lr 0.01 --momentum 0.9 --weight_decay 0.001 --optimiser sgd --num_epochs 40"

# tmux new-session -d -s yolo1 "python pretraining.py --gpu 1 --lr 0.01 --momentum 0.9 --weight_decay 0.001 --optimiser sgd"
# tmux new-session -d -s yolo2 "python pretraining.py --gpu 2 --lr 0.01 --momentum 0.9 --weight_decay 0.01 --optimiser sgd"

hparams = {
    "batch_size": 256,
    "num_epochs": args.num_epochs,
    "model": "YOLO",
    "dataset": "im64",
    "optimizer": args.optimiser,
    "learning_rate": args.lr,
    "momentum": args.momentum,
    "weight_decay": args.weight_decay,
    "gpu": args.gpu,
}

save_dir = f"lr{hparams['learning_rate']}_mom{hparams['momentum']}_wd{hparams['weight_decay']}_opt-{hparams['optimizer']}"
print(save_dir)

with open(f"{save_path}/dirs.sh", "a") as f:
    f.write(f"{save_dir} ")

# raise ValueError("Don't run this file.")

log = True
save = True
seed = 22
CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')

models = [
    PretrainingModel(in_channels=4, S=7).to(GPU),
]

for model in models:
    torch.manual_seed(seed)
    model.apply(reset_model)
if hparams["optimizer"] == "sgd": optimisers = [torch.optim.SGD(model.parameters(), lr=args.lr, momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True) for model in models]
elif hparams["optimizer"] == "adam": optimisers = [torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=hparams["weight_decay"]) for model in models]
else: raise ValueError("Invalid optimiser.")
loss_fns = [nn.CrossEntropyLoss() for _ in models]
# schedulers = [LR_Sched(optimiser, hparams["lr_schedule"]) for optimiser in optimisers]

if log:
    import wandb
    name = f"YOLO trial im64 wd={hparams['weight_decay']}"
    wandb.init(project="QuatLT23", name=name, config=hparams)
    for model in models:
        wandb.watch(model)

training_generator = torch.utils.data.DataLoader(Train(), batch_size=hparams["batch_size"], shuffle=True, num_workers=16, pin_memory=True)
print("Loading Validation data...")
validation_generator = torch.utils.data.DataLoader(Val(), batch_size=hparams["batch_size"], shuffle=True, num_workers=16, pin_memory=True)


num_epochs = hparams["num_epochs"]


# pretraining
for epoch in range(num_epochs):
    if epoch+1 in [11, 14, 17]:
        optimisers[0].param_groups[0]["lr"] *= 0.1
        print(f"reducing lr to {optimisers[0].param_groups[0]['lr']}")

    for batch_x, batch_y in tqdm(training_generator, desc = f"Epoch {epoch+1}/{num_epochs}", unit = "batch"):
        losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
        if log: wandb.log({f"loss Yolo": losses[i] for i, model in enumerate(models)})

    if save:
        for model in models:
            torch.save(model, f"{save_path}/{save_dir}/YOLO_pretraining_{epoch+1}.pt")

    if log:
        # accs = {
        #     train_accuracies(model, validation_generator, GPU)
        #     for model in models
        # }
        wandb.log(train_accuracies(models[0], validation_generator, GPU))


if log:
    wandb.finish()