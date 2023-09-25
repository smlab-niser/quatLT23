import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange

from utils.pruning import prune_model, reset_model
from utils.training import train_accuracy, train_multiple_models
from data_loaders.imagenet import Train, Val

model_choices = [
    "RN18_real",  "RN18_quat",
    "RN34_real",  "RN34_quat",
    "RN50_real",  "RN50_quat",
    "RN101_real", "RN101_quat",
    "RN152_real", "RN152_quat",
]


import argparse

parser = argparse.ArgumentParser(description="Train a model on ImageNet64x64.")
parser.add_argument("model_name", type=str, help="Name of the model.", choices=model_choices)
parser.add_argument("--save", action="store_true", help="Whether to save models.")
parser.add_argument("--log", action="store_true", help="Whether to log to wandb.")
parser.add_argument("--seed", type=int, help="Seed for initializing the model weights", default=21)
parser.add_argument("--gpu", type=int, choices=[0, 1, 2, 3], help="Which GPU to use.")

# tmux new-session -d -s YOLO "python pretraining.py --gpu 1 --lr 0.1 --momentum 0.9 --weight_decay 0.0001 --optimiser sgd --num_epochs 40"


args = parser.parse_args()
print(f"args = {args}")

if   args.model_name == "RN18_real":
    from models.resnet_real import ResNet18 as given_model
    if args.gpu is None: args.gpu = 0
elif args.model_name == "RN18_quat":
    from models.resnet_quat import ResNet18_quat as given_model
    if args.gpu is None: args.gpu = 1
elif args.model_name == "RN34_real":
    from models.resnet_real import ResNet34 as given_model
    if args.gpu is None: args.gpu = 0
elif args.model_name == "RN34_quat":
    from models.resnet_quat import ResNet34_quat as given_model
    if args.gpu is None: args.gpu = 1
elif args.model_name == "RN50_real":
    from models.resnet_real import ResNet50 as given_model
    if args.gpu is None: args.gpu = 0
elif args.model_name == "RN50_quat":
    from models.resnet_quat import ResNet50_quat as given_model
    if args.gpu is None: args.gpu = 1
elif args.model_name == "RN101_real":
    from models.resnet_real import ResNet101 as given_model
    if args.gpu is None: args.gpu = 0
elif args.model_name == "RN101_quat":
    from models.resnet_quat import ResNet101_quat as given_model
    if args.gpu is None: args.gpu = 1    
elif args.model_name == "RN152_real":
    from models.resnet_real import ResNet152 as given_model
    if args.gpu is None: args.gpu = 2
elif args.model_name == "RN152_quat":
    from models.resnet_quat import ResNet152_quat as given_model
    if args.gpu is None: args.gpu = 3
else:
    raise ValueError("Model name not recognised.")

save_path = f"saved_models/{args.model_name}_prune"

class LR_Sched:
    def __init__(self, optimiser, lr_schedule):
        self.optimiser = optimiser
        self.lr_schedule = lr_schedule

    def step(self, epoch):
        if epoch in self.lr_schedule:
            self.optimiser.param_groups[0]["lr"] = self.lr_schedule[epoch]
            # print(f"Learning rate changed to {self.optimiser.param_groups[0]['lr']}")


hparams = {
    "batch_size": 256,
    "num_epochs": 15,
    "num_prune": args.num_prune,
    "left_after_prune": args.left_after_prune,
    "model": args.model_name,
    "dataset": "imagenet64x64",
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "gpu": args.gpu,
    "lr_schedule": {
        0: 0.1,
        11: 0.01,
        13: 0.001,
    }
}


log = args.log
save = args.save
seed = args.seed
CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')

resume_from = 0
if args.resume:
    from os import listdir
    def rule(x):
        x = x[:-3].split("_")[-1]
        try: return int(x)
        except: return 0
    l = listdir(save_path)
    if len(l) < 2: raise ValueError("No saved models found.")
    # taking the second last save because the last save might not have finished all epochs
    last_full_save = sorted(l, key=rule, reverse=True)[1]
    models = [torch.load(f"{save_path}/{last_full_save}").to(GPU)]
    resume_from = rule(last_full_save)
    print(f"Resuming from {last_full_save}")
else:
    models = [
        given_model(name=args.model_name).to(GPU),
    ]

for model in models:
    torch.manual_seed(seed)
    model.apply(reset_model)
optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["lr_schedule"][0], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True) for model in models]
loss_fns = [nn.CrossEntropyLoss() for _ in models]
schedulers = [LR_Sched(optimiser, hparams["lr_schedule"]) for optimiser in optimisers]

if log:
    import wandb
    name = f"{args.model_name} prune" if not args.resume else f"{args.model_name} prune (resumed)"
    wandb.init(project="QuatLT23", name=name, config=hparams)
    for model in models:
        wandb.watch(model)

print("Loading Training data...")
training_generator = torch.utils.data.DataLoader(Train(), batch_size=hparams["batch_size"], shuffle=True)
print("Loading Validation data...")
validation_generator = torch.utils.data.DataLoader(Val(), batch_size=hparams["batch_size"], shuffle=False)


num_epochs = hparams["num_epochs"]


if not args.resume:
    # pretraining
    for epoch in trange(num_epochs, desc = "Pretraining", unit = "epoch"):
        
        for sched in schedulers:
            sched.step(epoch)

        for batch_x, batch_y in training_generator:
            losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
            if log: wandb.log({f"loss {model.name}": losses[i] for i, model in enumerate(models)})

        if save:
            for model in models:
                torch.save(model, f"{save_path}/{model.name}_unpruned.pt")

        # if log:
        #     accs = {
        #         f"test_acc {model.name}": train_accuracy(model, validation_generator, GPU) 
        #         for model in models
        #     }
        #     wandb.log(accs)


# pruning and retraining
for prune_it in range(resume_from, hparams["num_prune"]):

    models = [prune_model(model, 1-hparams["left_after_prune"]) for model in models]
    for model in models:
        torch.manual_seed(seed)
        model.apply(reset_model)

    optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["lr_schedule"][0], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True) for model in models]
    loss_fns = [nn.CrossEntropyLoss() for _ in models]
    schedulers = [LR_Sched(optimiser, hparams["lr_schedule"]) for optimiser in optimisers]


    for epoch in trange(num_epochs, desc=f"Pruning {prune_it+1}/{hparams['num_prune']}", unit = "epoch"):

        for sched in schedulers:
            sched.step(epoch)

        for batch_x, batch_y in training_generator:
            losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
            if log: wandb.log({f"loss {model.name}": losses[i] for i, model in enumerate(models)})

        if save:
            for model in models:
                torch.save(model, f"{save_path}/{model.name}_{prune_it+1}.pt")

        # if log:
        #     accs = {
        #         f"test_acc {model.name}": train_accuracy(model, validation_generator, GPU) 
        #         for model in models
        #     }
        #     wandb.log(accs)


if log:
    wandb.finish()