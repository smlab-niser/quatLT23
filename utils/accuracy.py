import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score,top_k_accuracy_score
import wandb
import os
    
#checking top5 test acc for a model
def top5acc_check(
    model_path,
    validation_generator,
    base_dir = "/home/aritra/project/quatLT23/3_imagenet_resnet/saved_models",
    GPU = torch.device("cuda")
):
    model = torch.load(f"{base_dir}/{model_path}.pth")
    accs5 = []
    j = 0
    for batch_x, batch_y in validation_generator:
        batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
        output = model(batch_x)
        acc5 = top_k_accuracy_score(batch_y.numpy(), output.detach().cpu().numpy(),k=5, labels=np.arange(1000))
        accs5.append(acc5*100)
        j += 1
        if j == 100: break
    test_acc5 = (np.array(accs5).mean())
    print(f"Test Accuracy (top5): {test_acc5:.2f}%")
    del output
    del batch_x, batch_y
    

#all accuracies for one model
def all_acc_model(
        model_path,
        training_generator,
        validation_generator,
        GPU = torch.device("cuda"),
        batches = np.inf
    ):
    model = torch.load(model_path)
    accs1 = []
    accs5 = []
    j = 0
    for batch_x, batch_y in tqdm(training_generator, desc="Train", unit="batch", total=min(batches, len(training_generator))):
        batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
        output = model(batch_x)
        acc1 = accuracy_score(batch_y.numpy(), output.argmax(1).cpu().numpy())
        accs1.append(acc1*100)
        acc5 = top_k_accuracy_score(batch_y.numpy(), output.detach().cpu().numpy(),k=5, labels=np.arange(1000))
        accs5.append(acc5*100)
        j += 1
        if j == batches: break
    train_acc1 = np.array(accs1).mean()
    train_acc5 = np.array(accs5).mean()

    accs1 = []
    accs5 = []
    j = 0
    for batch_x, batch_y in tqdm(validation_generator, desc="Val", unit="batch", total=min(batches, len(validation_generator))):
        batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
        output = model(batch_x)
        acc1 = accuracy_score(batch_y.numpy(), output.argmax(1).cpu().numpy())
        accs1.append(acc1*100)
        acc5 = top_k_accuracy_score(batch_y.numpy(), output.detach().cpu().numpy(),k=5, labels=np.arange(1000))
        accs5.append(acc5*100)
        j += 1
        if j == batches: break
    test_acc1 = np.array(accs1).mean()
    test_acc5=np.array(accs5).mean()
    
    return train_acc1, train_acc5, test_acc1, test_acc5

def all_acc_models(
        model_paths,
        training_generator,
        validation_generator,
        GPU = torch.device("cuda"),
        batches = np.inf
    ):
    acc_dict = {}
    j = 0
    for batch_x, batch_y in tqdm(training_generator, desc="Train", unit="batch", total=min(batches, len(training_generator))):
        batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
        for model_path in model_paths:
            model = torch.load(model_path)
            output = model(batch_x)
            acc1 = accuracy_score(batch_y.numpy(), output.argmax(1).cpu().numpy())
            acc5 = top_k_accuracy_score(batch_y.numpy(), output.detach().cpu().numpy(),k=5, labels=np.arange(1000))
            if j == 0: acc_dict[model_path] = [[acc1*100], [acc5*100]]
            else: 
                acc_dict[model_path][0].append(acc1*100)
                acc_dict[model_path][1].append(acc5*100)
        j += 1
        if j == batches: break
    acc_dict[model_path][0] = np.array(acc_dict[model_path][0]).mean()
    acc_dict[model_path][1] = np.array(acc_dict[model_path][1]).mean()
    del output
    del batch_x, batch_y
    
    j = 0
    for batch_x, batch_y in tqdm(validation_generator, desc="Train", unit="batch", total=min(batches, len(validation_generator))):
        batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
        for model_path in model_paths:
            model = torch.load(model_path)
            output = model(batch_x)
            acc1 = accuracy_score(batch_y.numpy(), output.argmax(1).cpu().numpy())
            acc5 = top_k_accuracy_score(batch_y.numpy(), output.detach().cpu().numpy(),k=5, labels=np.arange(1000))
            if j == 0: acc_dict[model_path].extend([[acc1*100], [acc5*100]])
            else: 
                acc_dict[model_path][2].append(acc1*100)
                acc_dict[model_path][3].append(acc5*100)
        j += 1
        if j == batches: break
    acc_dict[model_path][2] = np.array(acc_dict[model_path][2]).mean()
    acc_dict[model_path][3] = np.array(acc_dict[model_path][3]).mean()
    del output
    del batch_x, batch_y
    
    return acc_dict


def all_acc_prunemodels(
        models_folder,
        training_generator,
        validation_generator,
        GPU = torch.device("cuda"),
        batches = np.inf,
        save = None,
        log = False
    ):
    acc_dict = {}
    
    print("Loading folders and paths...")
    prunefolders = sorted(os.listdir(models_folder), key=lambda x: int(x))
    relevant_modelpaths = []
    for i in prunefolders:
        models = os.listdir(f"{models_folder}/{i}")
        models = sorted(models, key=lambda x: int(x.split(".pth")[0]))
        relevant_modelpaths.extend([f"{models_folder}/{i}/{model}" for model in models])
    
    print("Calculating train accuracies...")
    j = 0
    for batch_x, batch_y in tqdm(training_generator, desc="Train", unit="batch", total=min(batches, len(training_generator))):
        batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
        for model_path in relevant_modelpaths: 
            model = torch.load(model_path)
            output = model(batch_x)
            acc1 = accuracy_score(batch_y.numpy(), output.argmax(1).cpu().numpy())
            acc5 = top_k_accuracy_score(batch_y.numpy(), output.detach().cpu().numpy(),k=5, labels=np.arange(1000))
            if j == 0: acc_dict[model_path] = [[acc1*100], [acc5*100]]
            else: 
                acc_dict[model_path][0].append(acc1*100)
                acc_dict[model_path][1].append(acc5*100)
        j += 1
        if j == batches: break
    for model_path in relevant_modelpaths: 
      acc_dict[model_path][0] = np.array(acc_dict[model_path][0]).mean()
      acc_dict[model_path][1] = np.array(acc_dict[model_path][1]).mean()
    del output
    del batch_x, batch_y
    
    print("Calculating validation accuracies...")
    j = 0
    for batch_x, batch_y in tqdm(validation_generator, desc="Train", unit="batch", total=min(batches, len(validation_generator))):
        batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
        for model_path in relevant_modelpaths:
            model = torch.load(model_path)
            output = model(batch_x)
            acc1 = accuracy_score(batch_y.numpy(), output.argmax(1).cpu().numpy())
            acc5 = top_k_accuracy_score(batch_y.numpy(), output.detach().cpu().numpy(),k=5, labels=np.arange(1000))
            if j == 0: acc_dict[model_path].extend([[acc1*100], [acc5*100]])
            else: 
                acc_dict[model_path][2].append(acc1*100)
                acc_dict[model_path][3].append(acc5*100)
        j += 1
        if j == batches: break
    for model_path in relevant_modelpaths: 
        acc_dict[model_path][2] = np.array(acc_dict[model_path][2]).mean()
        acc_dict[model_path][3] = np.array(acc_dict[model_path][3]).mean()
    
    # print(acc_dict['/home/aritra/project/quatLT23/6_pruning/saved_models/RN18_real/4/1.pth'])
    del output
    del batch_x, batch_y
    
    if save:
        print(f"saving result as {save}...")
        with open(save, 'w') as f:
            f.write("Model Acc1_train Acc5_train Acc1_val Acc5_val\n")
            for model, acc in acc_dict.items():
                line = f"{model.split(models_folder)[-1]} {' '.join(str(x) for x in acc)}\n"
                f.write(line)
    acc_dict2 = {}
    
    for model in relevant_modelpaths:
        f = model.split("/")[-2]
        if f not in acc_dict2: acc_dict2[f] = [[i] for i in acc_dict[model]]
        else:
            for i in range(4):
                acc_dict2[f][i].append(acc_dict[model][i])
        
    return acc_dict2