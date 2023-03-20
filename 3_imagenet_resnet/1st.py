import numpy as np
import matplotlib.pyplot as plt
from utils import load_imagenet, one_hot
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
# from resnet import ResNet
import wandb

GPU = torch.device('cuda:0')
CPU = torch.device('cpu')


(x, y), (x_val, y_val) = load_imagenet()  # 5 parts of 10 parts
m = len(x)

hparams = {
    "batch_size": 64,
    "num_epochs": 40,
    "model": "resnet",
    "dataset": "imagenet",
    "optimizer": "adam",
    "learning_rate": 1e-3,  # 1.2e-5, 1.2e-7 can be tried
}

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.to(GPU)
optimiser = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])
losses = []

wandb.init(project="QuatLT23", name="resnet from pytorch", config=hparams)

wandb.watch(model)

batch_size = hparams["batch_size"]
num_epochs = hparams["num_epochs"]

for epoch in trange(num_epochs):

    for i in range(0, len(x), batch_size):
        batch_x, batch_y = x[i:i+batch_size].to(GPU), y[i:i+batch_size].to(GPU)
        optimiser.zero_grad()
        output = model(batch_x)
        loss = F.mse_loss(output, batch_y)
        loss.backward()
        optimiser.step()

    torch.save(model, "resnet_imagenet.pth")

    train_accuracy = accuracy_score(y[:5000].argmax(1),     model(x[:5000].to(GPU)).to(CPU).argmax(1))
    test_accuracy  = accuracy_score(y_val[:5000].argmax(1), model(x_val[:5000].to(GPU)).to(CPU).argmax(1))
    
    wandb.log({"train_accuracy": train_accuracy, "test_accuracy": test_accuracy, "loss": loss.item()})

    losses.append(loss.item())

wandb.finish()

torch.save(model, "resnet152_pytorch.pth")