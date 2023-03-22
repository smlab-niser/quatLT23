import numpy as np
from utils import load_imagenet
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
# import torchvision
import time
from resnet import ResNet152

hparams = {
    "batch_size": 256,
    "num_epochs": 40,
    "model": "resnet152 custom built",
    "dataset": "imagenet",
    "optimizer": "sgd",
    "learning_rate": 0.05,
    "gpu": 3,
}

GPU = torch.device(f'cuda:{hparams["gpu"]}')
CPU = torch.device('cpu')

log = True

model_save_name = f"B={hparams['batch_size']}_E={hparams['num_epochs']}_O={hparams['optimizer']}_ll={hparams['learning_rate']}.pth"

model = ResNet152()
model.to(GPU)
optimiser = torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"])

if log:
    import wandb
    wandb.init(project="QuatLT23", name="my ResNet152 run 2", config=hparams)
    wandb.watch(model)

print("Loading data...")
(x, y), (x_val, y_val) = load_imagenet()
m = len(x)

batch_size = hparams["batch_size"]
num_epochs = hparams["num_epochs"]

for epoch in range(num_epochs):

    t0 = time.time()

    pbar = tqdm(total=m//batch_size+1, desc=f"Epoch {epoch+1}/{num_epochs}")

    for i in range(0, len(x), batch_size):
        batch_x, batch_y = x[i:i+batch_size].to(GPU), y[i:i+batch_size].to(GPU)
        optimiser.zero_grad()
        output = model(batch_x)
        loss = F.mse_loss(output, batch_y)
        loss.backward()
        optimiser.step()

        pbar.update(1)

    t1 = time.time()

    torch.save(model, model_save_name)

    t2 = time.time()

    ratio = 0.005
    train_mask = np.random.choice(a=[False, True], size=m, p=[1-ratio, ratio])
    val_mask   = np.random.choice(a=[False, True], size=len(y_val), p=[1-ratio, ratio])
    model = model.to(CPU)
    train_accuracy = accuracy_score(y[train_mask].argmax(1),     model(x[train_mask]).argmax(1))
    test_accuracy  = accuracy_score(y_val[val_mask].argmax(1), model(x_val[val_mask]).argmax(1))
    model = model.to(GPU)

    t3 = time.time()

    if log: wandb.log(
        {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "loss": loss.item(),
            "training_time": t1-t0,
            "saving_time": t2-t1,
            "accuracy_time": t3-t2,
        }
    )

    pbar.close()

wandb.finish()
