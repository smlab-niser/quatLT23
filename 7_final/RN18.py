import torch
from torch import nn
from tqdm import tqdm

# from data_loaders.cifar100 import Train, Val
from data_loaders.ILSVRC import Train, Val
from models.resnet_real import ResNet18
from models.resnet_quat import ResNet18_quat
from utils.training import one_epoch
from utils.pruning import prune_model

hparams = {
    "batch_size": 256,
    "num_epochs": 15,
    "num_prune": 13,
    "prune_part": 0.8,
    "model": "ResNet18",
    "dataset": "ILSVRC",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "gpu": 0,
}

CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')

# # make and save the initiatiated pruned models
# save_dir = "saved_models/RN18_real"
# model_info = {}
# pruning_percentages = hparams["pruning_percentages"]
# for i in pruning_percentages:
#     prune_ratio = i/100
#     path = f"{save_dir}/{i}"
#     name = "0.pth"
#     model = prune_model(ResNet18(4).to(GPU), prune_ratio)
#     model_info[f"RN18_{i}%"] = {
#         "prune_ratio": prune_ratio,
#         "path": path,
#         "name": name,
#         "model": model,
#         "optimiser": torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"]),
#         "loss_fn": nn.CrossEntropyLoss(),
#     }

models = [ResNet18(4).to(GPU), ResNet18_quat(4).to(GPU)]
optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"]) for model in models]
loss_fns = [nn.CrossEntropyLoss() for _ in models]


log = False

loss_fn = nn.CrossEntropyLoss()

if log:
    import wandb
    wandb.init(project="QuatLT23", name="RN18 ILSVRC (all pruned together)", config=hparams)
    # wandb.watch(model)

print("Loading Train data...")
training_generator = torch.utils.data.DataLoader(Train(), shuffle=True, batch_size=256, num_workers=8)
# print("Loading Validation data...")
# validation_generator = torch.utils.data.DataLoader(Val(), shuffle=True, batch_size=256, num_workers=8)

num_epochs = hparams["num_epochs"]



def train_multiple_models(
        x, y,
        models,
        optimisers,
        loss_fns: list,
        GPU: torch.device = GPU
    ):
    losses = []
    for model, optimiser, loss_fn in zip(models, optimisers, loss_fns):
        loss = one_epoch(model, x, y, optimiser, loss_fn, GPU)
        losses.append(loss.item())
    return losses


# training once
print("Pretraining...")
for epoch in range(num_epochs):
    for batch_x, batch_y in tqdm(training_generator, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
        if log:
            wandb.log(
                {
                    "RN18_real": losses[0],
                    "RN18_quat": losses[1],
                }
            )

print("pretraining done... saving model")

torch.save(models[0], f"saved_models/RN18/real_unpruned.pth")
torch.save(models[1], f"saved_models/RN18/quat_unpruned.pth")


# pruning iteratively
for prune_it in range(hparams["num_prune"]):
    models = [prune_model(model, hparams["prune_part"]) for model in models]
    optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"]) for model in models]
    loss_fns = [nn.CrossEntropyLoss() for _ in models]

    for epoch in range(num_epochs):
        for batch_x, batch_y in tqdm(training_generator, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
            if log:
                wandb.log(
                    {
                        "RN18_real": losses[0],
                        "RN18_quat": losses[1],
                    }
                )

    print(f"pruning {prune_it+1}/{hparams['num_prune']} done, saving model", end="... ")

    torch.save(models[0], f"saved_models/RN18/real_prune{prune_it+1}.pth")
    torch.save(models[1], f"saved_models/RN18/quat_prune{prune_it+1}.pth")

    print("saved")


if log:
    wandb.finish()
