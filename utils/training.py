import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score,top_k_accuracy_score
import wandb

def train(
    model, num_epochs,
    training_generator,
    validation_generator,
    optimiser,
    loss_fn,
    save = None,
    GPU = torch.device("cuda"),
    log = False,
    epoch_shift=0,
    calculate_accuracy=True
):
    train_acc, test_acc = [], []
    for epoch in range(epoch_shift, num_epochs):
        if calculate_accuracy: print("Training")
        # j = 0
        for batch_x, batch_y in tqdm(training_generator, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            batch_x, batch_y = batch_x.to(GPU), F.one_hot(batch_y.long().flatten(), num_classes=1000).to(GPU)
            optimiser.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y.argmax(1))
            loss.backward()
            optimiser.step()
            # j += 1
            # if j == 20: break
        
        del output
        del batch_x, batch_y
        
        if calculate_accuracy:
            print("Calculating Accuracy")
            accs = []
            j = 0
            for batch_x, batch_y in training_generator:
                batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
                output = model(batch_x)
                acc = accuracy_score(batch_y.numpy(), output.argmax(1).cpu().numpy())
                accs.append(acc*100)
                j += 1
                if j == 100: break
            train_acc.append(np.array(accs).mean())
            print(f"Train Accuracy: {train_acc[-1]:.2f}%")

            del output
            del batch_x, batch_y

            accs = []
            j = 0
            for batch_x, batch_y in validation_generator:
                batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
                output = model(batch_x)
                acc = accuracy_score(batch_y.numpy(), output.argmax(1).cpu().numpy())
                accs.append(acc*100)
                j += 1
                if j == 100: break
            test_acc.append(np.array(accs).mean())
            print(f"Test Accuracy: {test_acc[-1]:.2f}%")
            
            del output
            del batch_x, batch_y

            if log: wandb.log(
                {
                    "train_acc": train_acc[-1],
                    "test_acc": test_acc[-1],
                    "loss": loss.item(),
                }
            )
        elif log: wandb.log(
            {
                "loss": loss.item(),
            }
        )

        if save:
            torch.save(model, f"saved_models/{save}_E={epoch+1}.pth")
            
    return train_acc, test_acc

#train and measure top1 and top5 train and test accuracy.
def train_multiacc(
    model, num_epochs,
    training_generator,
    validation_generator,
    optimiser,
    loss_fn,
    save = None,
    GPU = torch.device("cuda"),
    log = False,
    epoch_shift=0,
    top5acc = None
):
    train_acc1, test_acc1 = [], []
    train_acc5, test_acc5 = [], []
    for epoch in range(epoch_shift, num_epochs):
        print("Training")
        # j = 0
        for batch_x, batch_y in tqdm(training_generator, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            batch_x, batch_y = batch_x.to(GPU), F.one_hot(batch_y.long().flatten(), num_classes=1000).to(GPU)
            optimiser.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y.argmax(1))
            loss.backward()
            optimiser.step()
            # j += 1
            # if j == 20: break
        
        del output
        del batch_x, batch_y

        print("Calculating Accuracy")

        accs1 = []
        accs5 = []
        j = 0
        for batch_x, batch_y in training_generator:
            batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
            output = model(batch_x)
            acc1 = accuracy_score(batch_y.numpy(), output.argmax(1).cpu().numpy())
            accs1.append(acc1*100)
            if top5acc:
                acc5 = top_k_accuracy_score(batch_y.numpy(), output.detach().cpu().numpy(),k=5, labels=np.arange(1000))
                accs5.append(acc5*100)
            j += 1
            if j == 100: break
        train_acc1.append(np.array(accs1).mean())
        print(f"Train Accuracy (top1): {train_acc1[-1]:.2f}%")
        if top5acc:
            train_acc5.append(np.array(accs5).mean())
            print(f"Train Accuracy (top5): {train_acc5[-1]:.2f}%")
    
        del output
        del batch_x, batch_y

        accs1 = []
        accs5 = []
        j = 0
        for batch_x, batch_y in validation_generator:
            batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
            output = model(batch_x)
            acc1 = accuracy_score(batch_y.numpy(), output.argmax(1).cpu().numpy())
            accs1.append(acc1*100)
            if top5acc:
                acc5 = top_k_accuracy_score(batch_y.numpy(), output.detach().cpu().numpy(),k=5, labels=np.arange(1000))
                accs5.append(acc5*100)
            j += 1
            if j == 100: break
        test_acc1.append(np.array(accs1).mean())
        print(f"Test Accuracy (top1): {test_acc1[-1]:.2f}%")
        if top5acc:
            test_acc5.append(np.array(accs5).mean())
            print(f"Test Accuracy (top5): {test_acc5[-1]:.2f}%")
        del output
        del batch_x, batch_y
        if top5acc:
            if log: wandb.log(
                {
                    "train_acc1": train_acc1[-1],
                    "test_acc1": test_acc1[-1],
                    "train_acc5": train_acc5[-1],
                    "test_acc5": test_acc5[-1],
                    "loss": loss.item(),
                }
            )
        else:
            if log: wandb.log(
                {
                    "train_acc1": train_acc1[-1],
                    "test_acc1": test_acc1[-1],
                    "loss": loss.item(),
                }
            )

        if save:
            torch.save(model, f"saved_models/{save}_E={epoch}.pth")
    if top5acc:        
        return train_acc1, test_acc1, train_acc5, test_acc5
    else:
        return train_acc1, test_acc1
    
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