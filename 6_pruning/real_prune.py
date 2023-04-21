import torch
from torch import nn

from data_loaders.imagenet import Train, Val
from models.resnet_real import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils.training import train
from utils.pruning import prune_model

hparams = {
    "prune": None,
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

# save_name = f"4-new-{hparams['model']}_{hparams['dataset']}_B={hparams['batch_size']}_O={hparams['optimizer']}_ll={hparams['learning_rate']}"
save_name = f"(test)"

model =  ResNet18(4)

model.to(GPU)
optimiser = torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"])
loss_fn = nn.CrossEntropyLoss()

if log:
    import wandb
    wandb.init(project="QuatLT23", name="real_prune.py", config=hparams)
    wandb.watch(model)

print("Loading data...")
training_generator = torch.utils.data.DataLoader(Train(), shuffle=True, batch_size=hparams["batch_size"], num_workers=4)
validation_generator = torch.utils.data.DataLoader(Val(), shuffle=True, batch_size=hparams["batch_size"], num_workers=4)

num_epochs = hparams["num_epochs"]


if hparams["prune"]:
    model = prune_model(model, hparams["prune"])
# remove the pruned weights

train(model, num_epochs, training_generator, validation_generator, optimiser, loss_fn, GPU = GPU, log=True)

wandb.finish()
