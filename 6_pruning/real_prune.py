import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
import numpy as np
from data_loaders.imagenet import Train, Val
from models.resnet_real import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils import train

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
save_name = f"(test) real ResNet18 pruned to 25%"

model =  ResNet18(4)

model.to(GPU)
optimiser = torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"])
loss_fn = nn.CrossEntropyLoss()

if log:
    import wandb
    wandb.init(project="QuatLT23", name=save_name, config=hparams)
    wandb.watch(model)

print("Loading data...")
training_generator = torch.utils.data.DataLoader(Train(), shuffle=False, batch_size=hparams["batch_size"], num_workers=4)
validation_generator = torch.utils.data.DataLoader(Val(), shuffle=False, batch_size=hparams["batch_size"], num_workers=4)

num_epochs = hparams["num_epochs"]

# prune the model by 75%
def prune_model(model, fraction):
    """
    Prunes the weights of a PyTorch model based on the given pruning fraction.

    Args:
        model (nn.Module): PyTorch model to be pruned.
        fraction (float): Fraction of weights to be pruned. Value should be between 0 and 1.

    Returns:
        nn.Module: Pruned PyTorch model with trainable weights.
    """
    # Validate input fraction
    if fraction <= 0 or fraction >= 1:
        raise ValueError("Pruning fraction should be between 0 and 1, exclusive.")

    # Identify the pruning method to be used based on the model type
    if isinstance(model, nn.Module):
        prune_method = prune.l1_unstructured
    else:
        raise ValueError("The provided model is not a valid nn.Module.")

    # Iterate through each module in the model and apply pruning
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune_method(module, 'weight', amount=fraction)

    # Remove the pruned weights
    # prune.remove(model, 'weight')

    return model

model = prune_model(model, 0.75)
# remove the pruned weights

train(model, num_epochs, training_generator, validation_generator, optimiser, loss_fn, GPU = GPU, log=True)

wandb.finish()
