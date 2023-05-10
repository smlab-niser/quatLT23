import numpy as np
from data_loaders.ILSVRC import Train, Val
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from utils.training import train
from utils.accuracy import all_acc_models, top5acc_check
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
log = False #turn this on!

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

acc_dict = all_acc_models(
    relevant_models,
    training_generator,
    validation_generator,
    GPU = torch.device(f'cuda:0'),
    # batches=400
)

for model_n in acc_dict:
    train_accs1.append(acc_dict[model_n][0])
    train_accs5.append(acc_dict[model_n][1])
    test_accs1.append(acc_dict[model_n][2])
    test_accs1.append(acc_dict[model_n][3])
    if log: wandb.log(
        {
            "train_acc": acc_dict[model_n][0],
            "test_acc": acc_dict[model_n][2],
            "train_acc_top5": acc_dict[model_n][1],
            "test_acc_top5": acc_dict[model_n][3],
        }
    )

with open(f'acc_{m_start.split("_E")[0]}.txt', 'w') as f:
    for model, acc in acc_dict.items():
        line = f"{model} {' '.join(str(x) for x in acc)}\n"
        f.write(line)
# epochs = np.arange(1,len(relevant_models)+1)