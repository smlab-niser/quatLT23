import torch
# import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
from resnet_real import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from sklearn.metrics import accuracy_score
from pprint import pprint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',
                    required=True,
                    choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'],
                    help="The model to run.")
parser.add_argument('-s', '--sync', default=True, help="Whether to sync with wandb.")
args = parser.parse_args()
modelname = args.model
sync = bool(args.sync)

hparams = {
    "batch_size": 256,
    "num_epochs": 250,
    "model": modelname,
    "dataset": "cifar100_coarse",
    "num_classes": 20,
    "optimizer": "sgd",
    "learning_rate": 0.005,
    "device": "cuda:0",
}  # although not all are actually hyperparameters

pprint(hparams)

device = torch.device(hparams["device"])

train_file = f"../data/cifar100/coarse_train.csv" if hparams["dataset"] == "cifar100_coarse" else f"../data/cifar100/fine_train.csv"
test_file = f"../data/cifar100/coarse_test.csv" if hparams["dataset"] == "cifar100_coarse" else f"../data/cifar100/fine_test.csv"

train = pd.read_csv(train_file, header=None).values
x = torch.from_numpy(train[:, 1:].reshape(-1, 3, 32, 32)/255).float()
y = torch.nn.functional.one_hot(torch.from_numpy(train[:, 0]).long(), hparams["num_classes"])

test = pd.read_csv(test_file, header=None).values
x_test = torch.from_numpy(test[:, 1:].reshape(-1, 3, 32, 32)/255).float()
y_test = torch.nn.functional.one_hot(torch.from_numpy(test[:, 0]).long(), hparams["num_classes"])


if   hparams["model"] == "ResNet18" : model =  ResNet18(3, hparams["num_classes"])  # takes ~4.9s/epoch
elif hparams["model"] == "ResNet34" : model =  ResNet34(3, hparams["num_classes"])  # takes ~8.2s/epoch
elif hparams["model"] == "ResNet50" : model =  ResNet50(3, hparams["num_classes"])
elif hparams["model"] == "ResNet101": model = ResNet101(3, hparams["num_classes"])
elif hparams["model"] == "ResNet152": model = ResNet152(3, hparams["num_classes"])
else: raise ValueError("Invalid model name")

model.to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])
losses = []
train_acc, test_acc = [], []

wandb_name = f"{hparams['model']}_{hparams['dataset']}_B={hparams['batch_size']}_O={hparams['optimizer']}_ll={hparams['learning_rate']}"

if sync:
    import wandb
    wandb.init(project="QuatLT23", name=wandb_name, config=hparams)
    wandb.watch(model)

num_epochs = hparams["num_epochs"]
batch_size = hparams["batch_size"]

for epoch in trange(num_epochs):
    # print("Training")
    for i in range(0, len(x), batch_size):
        batch_x, batch_y = x[i:i+batch_size].to(device), y[i:i+batch_size].float().to(device)
        optimiser.zero_grad()
        output = model(batch_x)
        # print(f"{output.shape = }, {batch_y.shape = }")
        loss = F.mse_loss(output, batch_y)
        loss.backward()
        optimiser.step()
    losses.append(loss.item())

    train_accs, test_accs = [], []
    batch_size_acc = 1000

    # print("Calculating Accuracy")

    for i in range(0, len(x), batch_size_acc):
    # for i in range(0, 1000, batch_size_acc):
        batch_x, batch_y = x[i:i+batch_size_acc].to(device), y[i:i+batch_size_acc].numpy()
        train_pred = model(batch_x)
        acc = accuracy_score(batch_y.argmax(1), train_pred.argmax(1).cpu().numpy())
        train_accs.append(acc*100)
    train_acc.append(np.array(train_accs).mean())
    # print(f"Train Accuracy: {train_acc[-1]:.2f}%")

    for i in range(0, len(x_test), batch_size_acc):
    # for i in range(0, 1000, batch_size_acc):
        batch_x_test, batch_y_test = x_test[i:i+batch_size_acc].to(device), y_test[i:i+batch_size_acc].numpy()
        test_pred = model(batch_x_test)
        acc = accuracy_score(batch_y_test.argmax(1), test_pred.argmax(1).cpu().numpy())
        test_accs.append(acc*100)
    test_acc.append(np.array(test_accs).mean())
    # print(f"Test Accuracy: {test_acc[-1]:.2f}%")

    if sync: wandb.log(
        {
            "train_acc": train_acc[-1],
            "test_acc": test_acc[-1],
            "loss": loss.item(),
        }
    )

torch.save(model, f"saved_models/{wandb_name}_E={hparams['num_epochs']}.pth")
    
print(f"Final Train Accuracy: {train_acc[-1]:.2f}%")
print(f"Final Test Accuracy: {test_acc[-1]:.2f}%")

wandb.finish()