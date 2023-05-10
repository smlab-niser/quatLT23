import numpy as np
from data_loaders.ILSVRC import Train, Val
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from utils.training import train
from utils.accuracy import top5acc_check, all_acc_model
import os

hparams = {
    "batch_size": 64,
    "num_epochs": 34,
    "model": "ResNet152",
    "dataset": "ILSVRC",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "gpu": 0,
}

b_dir = "saved_models"
m_start = "ILSVRC_RN152_B256_E="
log = True

if log:
    import wandb
    wandb.init(project="QuatLT23", name="ILSVRC_RN152_B256 accuracies", config=hparams)

training_generator = torch.utils.data.DataLoader(Train(50000), shuffle=True, batch_size=hparams["batch_size"], num_workers=4)
validation_generator = torch.utils.data.DataLoader(Val(),      shuffle=True, batch_size=hparams["batch_size"], num_workers=4)

all_saved_models = os.listdir(b_dir)
relevant_models = [f"{b_dir}/{name}" for name in all_saved_models if name.startswith(m_start)]
relevant_models = sorted(relevant_models, key=lambda x: int(x.split("=")[-1][:-4]))

train_accs1 = []
train_accs5 = []
test_accs1 = []
test_accs5 = []

for i, model_path in enumerate(relevant_models):
    print(f"Model {i+1}/{len(relevant_models)}: {model_path}")
    tr1,tr5,te1,te5 = all_acc_model(
        model_path,
        training_generator,
        validation_generator,
        GPU = torch.device(f'cuda:0'),
        # batches=400
    )
    train_accs1.append(tr1)
    train_accs5.append(tr5)
    test_accs1.append(te1)
    test_accs5.append(te5)
    print(f"Train Accuracy: \t{tr1:.2f}%")
    print(f"Val Accuracy: \t\t{te1:.2f}%")
    print(f"Train Accuracy (top5): \t{tr5:.2f}%")
    print(f"Val Accuracy (top5): \t{te5:.2f}%\n")

    if log: wandb.log(
        {
            "train_acc": tr1,
            "test_acc": te1,
            "train_acc_top5": tr5,
            "test_acc_top5": te5,
        }
    )

# epochs = np.arange(1,len(relevant_models)+1)