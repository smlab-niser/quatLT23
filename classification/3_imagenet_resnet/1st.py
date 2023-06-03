import numpy as np
from utils import Imagenet64_train, Imagenet64_val, one_hot
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch import nn

hparams = {
    "batch_size": 256,
    "num_epochs": 40,
    "model": "ResNet152",
    "dataset": "imagenet",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "gpu": 0,
}

GPU = torch.device(f'cuda:{hparams["gpu"]}')
CPU = torch.device('cpu')

log = False

wandb_name = f"4-{hparams['model']}_{hparams['dataset']}_B={hparams['batch_size']}_O={hparams['optimizer']}_ll={hparams['learning_rate']}"


if   hparams["model"].lower() == "resnet18" : model =  ResNet18(4)
elif hparams["model"].lower() == "resnet34" : model =  ResNet34(4)
elif hparams["model"].lower() == "resnet50" : model =  ResNet50(4)
elif hparams["model"].lower() == "resnet101": model = ResNet101(4)
elif hparams["model"].lower() == "resnet152": model = ResNet152(4)
else: raise ValueError("Invalid model name")

# if   hparams["model"].lower() == "resnet18" : model =  resnet18 (4)
# elif hparams["model"].lower() == "resnet34" : model =  resnet34 (4)
# elif hparams["model"].lower() == "resnet50" : model =  resnet50 (4)
# elif hparams["model"].lower() == "resnet101": model =  resnet101(4)
# elif hparams["model"].lower() == "resnet152": model =  resnet152(4)
# else: raise ValueError("Invalid model name")


model.to(GPU)
optimiser = torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"])
# optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

if log:
    import wandb
    wandb.init(project="QuatLT23", name=wandb_name, config=hparams)
    wandb.watch(model)

print("Loading data...")
training_generator = torch.utils.data.DataLoader(Imagenet64_train(), batch_size=hparams["batch_size"], num_workers=4)
validation_generator = torch.utils.data.DataLoader(Imagenet64_val(), batch_size=450, num_workers=4)
# m = len(training_set)

batch_size = hparams["batch_size"]
num_epochs = hparams["num_epochs"]

train_acc, test_acc = [], []

for epoch in range(num_epochs):
    print("Training")
    for batch_x, batch_y in tqdm(training_generator):
        # print("Batching")
        # print(f"{batch_x.shape = }, {batch_y.shape = }")
        # print(batch_y.max(), batch_y.min())
        batch_x, batch_y = batch_x.to(GPU), F.one_hot(batch_y.long().flatten(), num_classes=1000).to(GPU)
        optimiser.zero_grad()
        output = model(batch_x)
        # print(f"{output.shape = }, {batch_y.shape = }")
        loss = loss_fn(output, batch_y.argmax(1))
        loss.backward()
        optimiser.step()
    # losses.append(loss.item())

    print("Calculating Accuracy")

    train_accs, test_accs = [], []

    j = 0
    for batch_x, batch_y in training_generator:
        batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
        train_pred = model(batch_x)
        acc = accuracy_score(batch_y.numpy(), train_pred.argmax(1).cpu().numpy())
        train_accs.append(acc*100)
        j += 1
        if j == 100: break
    train_acc.append(np.array(train_accs).mean())
    print(f"Train Accuracy: {train_acc[-1]:.2f}%")


    j = 0
    for batch_x, batch_y in validation_generator:
        batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
        test_pred = model(batch_x)
        acc = accuracy_score(batch_y.numpy(), test_pred.argmax(1).cpu().numpy())
        train_accs.append(acc*100)
        j += 1
        if j == 100: break
    test_acc.append(np.array(train_accs).mean())
    
    # for i in range(0, len(x_val), batch_size_acc):
    # for i in trange(0, 45000, batch_size_acc):
    #     batch_x_test, batch_y_test = x_val[i:i+batch_size_acc].to(GPU), y_val[i:i+batch_size_acc].numpy()
    #     test_pred = model(batch_x_test)
    #     acc = accuracy_score(batch_y_test.argmax(1), test_pred.argmax(1).cpu().numpy())
    #     test_accs.append(acc*100)
    # test_acc.append(np.array(test_accs).mean())
    print(f"Test Accuracy: {test_acc[-1]:.2f}%")

    if log: wandb.log(
        {
            "train_acc": train_acc[-1],
            "test_acc": test_acc[-1],
            "loss": loss.item(),
        }
    )
    
    torch.save(model, f"saved_models/{wandb_name}_E={hparams['num_epochs']}.pth")

if log: wandb.finish()
