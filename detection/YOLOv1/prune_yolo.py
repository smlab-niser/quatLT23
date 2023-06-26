import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from utils.pruning import prune_model, reset_model
from utils.training import train_accuracies, train_multiple_models
from data_loaders.pascal_voc import PascalVOC
# from models.yolo_real import Pretraining as Real
# from models.yolo_quat import Pretraining as Quat
from models.yolo_real import Yolov1 as Real
from models.yolo_quat import Yolov1 as Quat
from utils.yolo_utils import YoloLoss, YoloSceduler
import torchvision.transforms.functional as F
from torchvision import transforms

save_path = f"saved_models/prune"

hparams = {
    "batch_size": 64,
    "num_epochs": 80,
    "model": "YOLO",
    "dataset": "Pascal VOC",
    "optimizer": "sdg",
    "learning_rate": 0.5,  # does not matter, will be changed by scheduler
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "num_prune": 25,
    "left_after_prune": 0.7,
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
    # Real(in_channels=4, name="real", base_path="/home/aritra/project/quatLT23/detection/YOLOv1/saved_models/pretrain_real/18.pt").to(GPU),
    Quat(in_channels=4, name="quat", base_path="/home/aritra/project/quatLT23/detection/YOLOv1/saved_models/pretrain_quat/18.pt").to(GPU),
]

optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"]) for model in models]
schedulers = [YoloSceduler(optimiser) for optimiser in optimisers]
loss_fns = [YoloLoss() for _ in models]

if log:
    import wandb
    name = f"YOLO prune quat"
    wandb.init(project="QuatLT23", name=name, config=hparams)
    for model in models:
        wandb.watch(model)

print("Loading Training Data")
training_generator = DataLoader(PascalVOC("train"), batch_size=hparams["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
print("Loading Validation Data")
val_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    lambda img: torch.cat((img, F.rgb_to_grayscale(img)), dim=0),
])
validation_generator = DataLoader(PascalVOC("val", transform=val_transform), batch_size=hparams["batch_size"], shuffle=True, num_workers=8, pin_memory=True)

num_epochs = hparams["num_epochs"]


# Unpruned Training
for epoch in trange(num_epochs, desc = "Unpruned Training", unit = "epoch"):

    for scheduler in schedulers:
        scheduler.change()

    train_losses = []
    for batch_x, batch_y in training_generator:
        losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
        train_losses.append(losses)
        # if log: wandb.log({f"loss Yolo {model.name}": losses[i] for i, model in enumerate(models)})
    train_losses = np.array(train_losses).mean(axis=0)

    test_losses = []
    for batch_x, batch_y in validation_generator:
        outputs = [model(batch_x.to(GPU)) for model in models]
        test_losses.append([loss_fn(output, batch_y.flatten().to(GPU)).item() for output, loss_fn in zip(outputs, loss_fns)])
    test_losses = np.array(test_losses).mean(axis=0)

    if save and epoch%5 == 0:
        for model in models:
            torch.save(model, f"{save_path}_{model.name}/unpruned.pt")

    if log:
        dat = {f"train loss Yolo {model.name}": train_losses[i] for i, model in enumerate(models)} | {f"test loss Yolo {model.name}": test_losses[i] for i, model in enumerate(models)}
        # print(dat)
        wandb.log(dat)

    # if log: wandb.log(train_accuracies(models[0], validation_generator, GPU, name=models[0].name) | train_accuracies(models[1], validation_generator, GPU, name=models[1].name))




# Pruning
for prune_it in range(hparams["num_prune"]):
    
    # prune
    models = [prune_model(model, 1-hparams["left_after_prune"]) for model in models]
    
    # reset everything
    for model in models: model.reset()
    optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"]) for model in models]
    loss_fns = [YoloLoss() for _ in models]
    for i, scheduler in enumerate(schedulers): scheduler.reset(optimisers[i])

    # train
    for epoch in trange(num_epochs, desc = f"Prune {prune_it+1}/{hparams['num_prune']}", unit = "epoch"):

        for scheduler in schedulers:
            scheduler.change()

        train_losses = []
        for batch_x, batch_y in tqdm(training_generator, desc = f"Epoch {epoch+1}/{num_epochs}", unit = "batch"):
            losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
            train_losses.append(losses)
            # if log: wandb.log({f"loss Yolo {model.name}": losses[i] for i, model in enumerate(models)})
        train_losses = np.array(train_losses).mean(axis=0)

        test_losses = []
        for batch_x, batch_y in validation_generator:
            outputs = [model(batch_x.to(GPU)) for model in models]
            test_losses.append([loss_fn(output, batch_y.flatten().to(GPU)).item() for output, loss_fn in zip(outputs, loss_fns)])
        test_losses = np.array(test_losses).mean(axis=0)

        if save and epoch%5 == 0:
            for model in models:
                torch.save(model, f"{save_path}_{model.name}/{prune_it+1}.pt")

        if log:
            dat = {f"train loss Yolo {model.name}": train_losses[i] for i, model in enumerate(models)} | {f"test loss Yolo {model.name}": test_losses[i] for i, model in enumerate(models)}
            # print(dat)
            wandb.log(dat)


if log:
    wandb.finish()