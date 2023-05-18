import torch
from torch import nn
from tqdm import trange

# from data_loaders.cifar100 import Train, Val
from data_loaders.cifar100 import Train, Val
from models.resnet_real import ResNet18
from models.resnet_quat import ResNet18_quat
from utils.training import train_accuracy, train_multiple_models
from torch.optim.lr_scheduler import LambdaLR as LRS
# from torch.optim.lr_scheduler import CyclicLRWithRestarts as LRS
# from utils.pruning import reset_model
import math
import torchvision
from torchvision import transforms

hparams = {
    "batch_size": 100,
    "num_epochs": 120,
    "warmup_epochs": 10,
    "model": "ResNet18",
    "dataset": "cifar100",
    "version": "fine",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "gpu": 0,
}



def adjust_learning_rate(epoch):
    batch_idx = 0
    data_nums = 50000//hparams["batch_size"]
    type="cosine"
    if epoch < hparams["warmup_epochs"]:
        epoch += float(batch_idx + 1) / data_nums
        lr_adj = 1. * (epoch / hparams["warmup_epochs"])
    elif type == "linear":
        if epoch < 20 + hparams["warmup_epochs"]:
            lr_adj = 1.
        elif epoch < 50 + hparams["warmup_epochs"]: # warmup_epochs = 10
            lr_adj = 1e-1
        elif epoch < 80 + hparams["warmup_epochs"]:
            lr_adj = 1e-2
        else:
            lr_adj = 1e-3
    elif type == "cosine":
        run_epochs = epoch - hparams["warmup_epochs"]
        total_epochs = hparams["num_epochs"] - hparams["warmup_epochs"]
        T_cur = float(run_epochs * data_nums) + batch_idx
        T_total = float(total_epochs * data_nums)
        lr_adj = 0.5 * (1 + math.cos(math.pi * T_cur / T_total))
        
    return lr_adj






log = True
save = False
seed = 21
save_path = "saved_models/RN18"
CPU = torch.device('cpu')
GPU = torch.device(f'cuda:{hparams["gpu"]}')
num_classes = 20 if hparams["version"] == "coarse" else 100

models = [
    ResNet18(num_classes=num_classes).to(GPU),      # real
    # ResNet18(4, num_classes=num_classes).to(GPU)  # quat
]
# for model in models:
#     torch.manual_seed(seed)
#     model.apply(reset_model)
optimisers = [torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True) for model in models]
loss_fns = [nn.CrossEntropyLoss() for _ in models]


if log:
    import wandb
    wandb.init(project="QuatLT23", name="RN18 cifar100 last try for today", config=hparams)
    wandb.watch(models[0])
    # wandb.watch(models[1])

# print("Loading Train data...")
# training_generator = torch.utils.data.DataLoader(Train(version=hparams["version"]), shuffle=True, batch_size=hparams["batch_size"], num_workers=8)
# print("Loading Validation data...")
# validation_generator = torch.utils.data.DataLoader(Val(version=hparams["version"]), shuffle=True, batch_size=hparams["batch_size"], num_workers=8)

transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root='/home/aritra/project/quatLT23/9_cifar100_RN18/cifar100', train=True, download=True, transform=transform_train)
training_generator = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, drop_last=True)

testset = torchvision.datasets.CIFAR100(root='/home/aritra/project/quatLT23/9_cifar100_RN18/cifar100', train=False, download=True, transform=transform_test)
validation_generator = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, drop_last=True)



num_epochs = hparams["num_epochs"]


scheduler = [LRS(
        optimiser,
        lr_lambda=adjust_learning_rate
    ) for optimiser in optimisers]

# scheduler = [LRS(
#         optimiser,
#         lr_lambda=adjust_learning_rate
#     ) for optimiser in optimisers]


for epoch in trange(num_epochs, desc="Training"):
    for batch_x, batch_y in training_generator:
        losses = train_multiple_models(batch_x, batch_y, models, optimisers, loss_fns, GPU)
        for sched in scheduler:
            sched.step()
        if log:
            wandb.log(
                {
                    "loss RN18_real": losses[0],
                    # "loss RN18_quat": losses[1],
                }
            )

    real_test_acc = train_accuracy(models[0], validation_generator, GPU)
    # quat_test_acc = train_accuracy(models[1], validation_generator, GPU)
    if log: wandb.log({"test_acc RN18_real": real_test_acc})


if log:
    wandb.finish()