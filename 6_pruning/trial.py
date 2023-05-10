import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
import numpy as np
from data_loaders.ILSVRC import Train, Val
from models.resnet_real import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils.training import train
import os

model_name = "RN18_real"
models_folder = f"6_pruning/saved_models/{model_name}"

acc_dict = {}
    
prunefolders = sorted(os.listdir(models_folder), key=lambda x: int(x))
relevant_models = []
for i in prunefolders:
    models = os.listdir(f"{models_folder}/{i}")
    models = sorted(models, key=lambda x: int(x.split(".pth")[0]))
    relevant_models.extend([f"{models_folder}/{i}/{model}" for model in models])

for model in relevant_models:
    acc_dict[model] = np.arange(1,5,1)
    
# with open(f'acc_{models_folder.split("/")[-1]}_prunes.txt', 'w') as f:
#     f.write("Model Acc1_train Acc5_train Acc1_val Acc5_val\n")
#     for model, acc in acc_dict.items():
#         line = f"{model.split(models_folder)[-1]} {' '.join(str(x) for x in acc)}\n"
#         f.write(line)

acc_dict2 = {}
for model in relevant_models:
    f,m = model.split("/")[-2:]
    print(f,m)
    if f not in acc_dict2: acc_dict2[f] = [[i] for i in acc_dict[model]]
    else:
        for i in range(4):
            acc_dict2[f][i].append(acc_dict[model][i])

    
    

