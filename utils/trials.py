# import os

# models_folder = "6_pruning/saved_models/RN18_real"

# print("Loading folders and paths...")
# prunefolders = sorted(os.listdir(models_folder), key=lambda x: int(x))
# relevant_modelpaths = []
# for i in prunefolders:
#     models = os.listdir(f"{models_folder}/{i}")
#     models = sorted(models, key=lambda x: int(x.split(".pth")[0]))
#     relevant_modelpaths.extend([f"{models_folder}/{i}/{model}" for model in models])
    
# print(relevant_modelpaths)

import numpy as np
a =[[[12.5], [22.65625], [26.953125], [29.6875], [36.71875], [31.640625], [37.109375], [41.40625], [46.09375], [41.015625], [43.359375], [47.265625], [42.96875], [44.921875], 43.359375], [[28.515625], [37.109375], [45.3125], [48.828125], [55.46875], [53.90625], [60.15625], [62.109375], [64.453125], [63.28125], [62.5], [66.015625], [67.1875], [66.796875], 65.625], [[15.234375], [25.0], [32.03125], [37.109375], [39.453125], [43.75], [41.40625], [42.1875], [41.015625], [43.75], [41.796875], [43.359375], [41.40625], [41.015625], 40.234375], [[34.765625], [48.4375], [60.9375], [59.765625], [64.453125], [69.53125], [66.40625], [66.796875], [69.140625], [67.578125], [69.53125], [71.875], [72.65625], [70.3125], 71.875]]

# print(np.array(a))

import wandb

hparams = {
    "batch_size": 256,
    "num_epochs": 15,
    "model": "ResNet18",
    "dataset": "ILSVRC",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "gpu": 0,
    "pruning_percentages": [100, 80, 64, 50, 32, 25, 20, 16, 13, 10, 8, 6, 4],
    "batches" : 1
}

model_name = "RN18_real"
models_folder = f"/home/aritra/project/quatLT23/6_pruning/saved_models/{model_name}"
save_name = f"acc_{model_name}_prunes_trial2.txt"
log = True #turn this on!

if log:
    import wandb
    wandb.init(project="QuatLT23", name="work_trial_DELETE", config=hparams)
    
a = {'1':[[1,2,3,4],[5,6,7,8]], '2':[[9,10,11,12],[13,14,15,16]]}

if log:
    wandb.log(
            {
                "train_acc_pruning": [a[i][0][-1] for i in a],
                "train_acc_top5_pruning": [a[i][1][-1] for i in a]}
    )
    for prune,accs in a.items():
        wandb.log(
            {
                f"{prune}%_train_acc": a[prune][0],
                f"{prune}%_train_acc_top5": a[prune][1]
            }
        )