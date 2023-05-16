import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


from data_loaders.mnist import Train, Val
from models.lenet_300_100 import Real, Quat
from utils.training_mnist import one_epoch, train_accuracy
from utils.pruning import prune_model, reset_model

hparams = {
    "batch_size": 64,
    "num_epochs": 15,
    "num_prune": 25,
    "prune_part": 1 - 0.8,
    "model": "Lenet_300_100",
    "dataset": "mnist",
    "optimizer": "adam",
    "learning_rate": 1.2e-3,
    "gpu": 0,
}

CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')


models = [Real().to(GPU), Quat().to(GPU)]
optimisers = [torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"]) for model in models]
loss_fns = [F.mse_loss for _ in models]


log = True
save = True

if log:
    import wandb
    wandb.init(project="QuatLT23", name="Lenet mnist (all pruned together)", config=hparams)
    # wandb.watch(model)

print("Loading Train data...")
training_generator = torch.utils.data.DataLoader(Train(), shuffle=True, batch_size=hparams["batch_size"], num_workers=4)
print("Loading Validation data...")
validation_generator = torch.utils.data.DataLoader(Val(), shuffle=True, batch_size=hparams["batch_size"], num_workers=4)

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
for epoch in tqdm(range(num_epochs), desc="Pretraining", unit="epoch"):
    for batch_x, batch_y in training_generator:
        losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
        if log:
            wandb.log(
                {
                    "loss Lenet_real": losses[0],
                    "loss Lenet_quat": losses[1],
                }
            )

    train_acc = train_accuracy(models[0], training_generator, GPU)
    test_acc = train_accuracy(models[0], validation_generator, GPU)
    if log: wandb.log({"Real train_acc": train_acc,"Real test_acc": test_acc})
    
    train_acc = train_accuracy(models[1], training_generator, GPU)
    test_acc = train_accuracy(models[1], validation_generator, GPU)
    if log: wandb.log({"Quat train_acc": train_acc,"Quat test_acc": test_acc})

print("pretraining done", end="...")

if save:
    torch.save(models[0], f"saved_models/Lenet/real_unpruned.pth")
    torch.save(models[1], f"saved_models/Lenet/quat_unpruned.pth")
    print("saved!")
else:
    print("not saving!")

# pruning iteratively
for prune_it in range(hparams["num_prune"]):
    models = [prune_model(model, hparams["prune_part"]) for model in models]
    optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"]) for model in models]
    loss_fns = [nn.CrossEntropyLoss() for _ in models]

    for epoch in tqdm(range(num_epochs), desc=f"Pruning {prune_it+1}/{hparams['num_prune']}", unit="epoch"):
        for batch_x, batch_y in training_generator:
            losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
            if log:
                wandb.log(
                    {
                        "Lenet_real": losses[0],
                        "Lenet_quat": losses[1],
                    }
                )

        train_acc = train_accuracy(models[0], training_generator, GPU)
        test_acc = train_accuracy(models[0], validation_generator, GPU)
        if log: wandb.log({"Real train_acc": train_acc,"Real test_acc": test_acc})

        train_acc = train_accuracy(models[1], training_generator, GPU)
        test_acc = train_accuracy(models[1], validation_generator, GPU)
        if log: wandb.log({"Quat train_acc": train_acc,"Quat test_acc": test_acc})

    print(f"pruning {prune_it+1}/{hparams['num_prune']} done", end="...")

    if save:
        torch.save(models[0], f"saved_models/Lenet/real_prune{prune_it+1}.pth")
        torch.save(models[1], f"saved_models/Lenet/quat_prune{prune_it+1}.pth")
        print("saved!")
    else:
        print("not saving!")


if log:
    wandb.finish()
