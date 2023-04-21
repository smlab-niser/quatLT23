import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange
import numpy as np
import wandb


def load_imagenet64(n=10):
    base_dir = "/mnt/data/datasets/imagenet/64x64/data"

    train_x_files = [f"{base_dir}/train/x{i}.npy" for i in range(1, 11)]
    train_y_files = [f"{base_dir}/train/y{i}.npy" for i in range(1, 11)]

    x_train, y_train = None, None

    for i in trange(n):
        if x_train is None:
            x_train = np.load(train_x_files[i])
            y_train = np.load(train_y_files[i])
        else:
            x_train = np.append(x_train, np.load(train_x_files[i]), axis=0)
            y_train = np.append(y_train, np.load(train_y_files[i]), axis=0)
    print("train data loaded")

    val_file = base_dir + "/val/val_data"
    val = np.load(val_file, allow_pickle=True)
    x_val = val["data"]
    y_val = np.array(val["labels"])
    del val
    print("val data loaded")

    x_train = torch.from_numpy(x_train.reshape(-1, 3, 64, 64)).float()
    y_train = torch.from_numpy(one_hot(y_train)).float()

    print("train data converted to tensors")

    x_val = torch.from_numpy(x_val.reshape(-1, 3, 64, 64)).float()
    y_val = torch.from_numpy(one_hot(y_val)).float()

    return (x_train, y_train), (x_val, y_val)

def train(
    model, num_epochs,
    training_generator,
    validation_generator,
    optimiser,
    loss_fn,
    save = None,
    GPU = torch.device("cuda"),
    log = False
):
    train_acc, test_acc = [], []
    for epoch in range(num_epochs):
        print("Training")
        for batch_x, batch_y in tqdm(training_generator, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            batch_x, batch_y = batch_x.to(GPU), F.one_hot(batch_y.long().flatten(), num_classes=1000).to(GPU)
            optimiser.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y.argmax(1))
            loss.backward()
            optimiser.step()
        
        del output

        print("Calculating Accuracy")

        train_accs, test_accs = [], []

        j = 0
        for batch_x, batch_y in training_generator:
            batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
            train_pred = model(batch_x)
            acc = accuracy_score(batch_y.numpy(), train_pred.argmax(1).cpu().numpy())
            train_accs.append(acc*100)
            j += 1
            if j == 100: break
        train_acc.append(np.array(train_accs).mean())
        print(f"Train Accuracy: {train_acc[-1]:.2f}%")


        j = 0
        for batch_x, batch_y in validation_generator:
            batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
            test_pred = model(batch_x)
            acc = accuracy_score(batch_y.numpy(), test_pred.argmax(1).cpu().numpy())
            train_accs.append(acc*100)
            j += 1
            if j == 100: break
        test_acc.append(np.array(train_accs).mean())
        print(f"Test Accuracy: {test_acc[-1]:.2f}%")

        if log: wandb.log(
            {
                "train_acc": train_acc[-1],
                "test_acc": test_acc[-1],
                "loss": loss.item(),
            }
        )

        if save:
            torch.save(model, f"saved_models/{save}_E={epoch}.pth")
    return train_acc, test_acc


def get_model_sparsity(model: nn.Module):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)  # all trainable parameters

    pruned_params = 0
    for name, buffer in model.named_buffers():
        if name[-5:] == '_mask':
            pruned_params += torch.sum(buffer == 0).item()

    sparsity = 100 * (total - pruned_params) / total
    return sparsity


def prune_model(model: nn.Module, pruning_percentage: float) -> nn.Module:
    """
    Prunes the weights of a PyTorch model based on the given pruning percentage.

    Args:
        model (nn.Module): PyTorch model to be pruned.
        pruning_percentage (float): Percentage of weights to be pruned. Value should be between 0 and 1.

    Returns:
        nn.Module: Pruned PyTorch model with trainable weights.
    """
    # Validate input pruning_percentage
    if pruning_percentage <= 0 or pruning_percentage >= 1:
        raise ValueError("Pruning percentage should be between 0 and 1, exclusive.")

    # Identify the pruning method to be used based on the model type
    if isinstance(model, nn.Module):
        prune_method = prune.l1_unstructured
    else:
        raise ValueError("The provided model is not a valid nn.Module.")

    # Iterate through each module in the model and apply pruning
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune_method(module, 'weight', amount=pruning_percentage)

    # Remove the pruned weights
    prune.remove(model, 'weight')

    return model, get_model_sparsity(model)
