import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from utils.pruning import prune_model, reset_model
from utils.training import train_multiple_models#, train_accuracies
from data_loaders.pascal_voc import PascalVOC
# from models.yolo_real import Pretraining as Real
# from models.yolo_quat import Pretraining as Quat
from models.yolo_real import Yolov1 as Real
from models.yolo_quat import Yolov1 as Quat
from utils.yolo_utils import YoloLoss, YoloSceduler, mAP
import torchvision.transforms.functional as F
from torchvision import transforms

save_path = f"saved_models/train"

hparams = {
    "batch_size": 64,
    "num_epochs": 140,
    "model": "YOLO",
    "dataset": "Pascal VOC",
    "optimizer": "adam",
    "learning_rate": 1e-2 * 1e-3,  # (don't change) + (change)
    # "momentum": 0.9,
    "weight_decay": 5e-4,
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

if hparams["optimizer"].lower() == "sgd": optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"]) for model in models]
elif hparams["optimizer"].lower() == "adam": optimisers = [torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"]) for model in models]
else: raise ValueError(f"Optimizer {hparams['optimizer']} not recognised")
schedulers = [YoloSceduler(optimiser, reduction=hparams["learning_rate"]) for optimiser in optimisers]  # , dropat=[17, 20, 25]
loss_fns = [YoloLoss() for _ in models]

if log:
    import wandb
    name = f"YOLO quat train final"
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

for epoch in range(num_epochs):

    for scheduler in schedulers:
        scheduler.change()

    train_losses = []
    for batch_x, batch_y in tqdm(training_generator, desc = f"Epoch {epoch+1}/{num_epochs}", unit = "batch"):
        losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
        train_losses.append(losses)
        dat = {f"loss Yolo {model.name}": losses[i] for i, model in enumerate(models)}
        # if log: wandb.log(dat)
        # else: print(dat)
    train_losses = np.array(train_losses).mean(axis=0)

    test_losses = []
    for batch_x, batch_y in validation_generator:
        outputs = [model(batch_x.to(GPU)) for model in models]
        test_losses.append([loss_fn(output, batch_y.flatten().to(GPU)).item() for output, loss_fn in zip(outputs, loss_fns)])
    test_losses = np.array(test_losses).mean(axis=0)

    if save and (epoch+1) in [60, 80, 120, 140]:
        for model in models:
            torch.save(model, f"{save_path}_{model.name}/{epoch+1}.pt")

    dat  = {f"train loss Yolo {model.name}": train_losses[i] for i, model in enumerate(models)}
    dat |= {f"test loss Yolo {model.name}": test_losses[i] for i, model in enumerate(models)}
    dat |= {f"train mAP Yolo {model.name}": mAP(model, training_generator, GPU) for model in models}
    dat |= {f"test mAP Yolo {model.name}": mAP(model, validation_generator, GPU) for model in models}
    if log: wandb.log(dat)
    else: print(dat)


if log:
    wandb.finish()