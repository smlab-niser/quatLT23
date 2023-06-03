import numpy as np
from data_loaders.ILSVRC import Train, Val
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from utils.training import train
from utils.accuracy import all_acc_prunemodels, top5acc_check, all_acc_models
hparams = {
    "batch_size": 256,
    "num_epochs": 15,
    "model": "ResNet18",
    "dataset": "ILSVRC",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "gpu": 0,
    "pruning_percentages": [100, 80, 64, 50, 32, 25, 20, 16, 13, 10, 8, 6, 4],
    "batches" : 400
}

model_name = "RN18_real"
models_folder = f"/home/aritra/project/quatLT23/6_pruning/saved_models/{model_name}"
save_name = f"acc_{model_name}_prunes.txt"
log = False # logging issues, will fix later

if log:
    import wandb
    wandb.init(project="QuatLT23", name="ILSVRC_RN18_realprunes_acc(400)", config=hparams)

training_generator = torch.utils.data.DataLoader(Train(50000), shuffle=True, batch_size=hparams["batch_size"], num_workers=4)
validation_generator = torch.utils.data.DataLoader(Val(),      shuffle=True, batch_size=hparams["batch_size"], num_workers=4)

acc_dict = all_acc_prunemodels(
    models_folder,
    training_generator,
    validation_generator,
    GPU = torch.device(f'cuda:0'),
    save = save_name,
    batches=hparams["batches"]
)

# print(acc_dict['100'])

# print(np.array(acc_dict["/6/3.pth"]))
if log:
    for i in range(hparams["num_epochs"]):
        for prune,accs in acc_dict.items():
            wandb.log(
                {
                    f"{prune}%_train_acc": acc_dict[prune][0][i],
                    f"{prune}%_test_acc": acc_dict[prune][2][i],
                    f"{prune}%_train_acc_top5": acc_dict[prune][1][i],
                    f"{prune}%_test_acc_top5": acc_dict[prune][3][i]
                }
            )
    for prune in acc_dict:
        wandb.log(
            {
                "prune ratio": int(prune),
                "train_acc_pruning": acc_dict[prune][0][-1],
                "test_acc_pruning": acc_dict[prune][2][-1],
                "train_acc_top5_pruning": acc_dict[prune][1][-1],
                "test_acc_top5_pruning": acc_dict[prune][3][-1]
            }
        )
wandb.finish()

# epochs = np.arange(1,len(relevant_models)+1)