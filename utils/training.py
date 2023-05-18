import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score,top_k_accuracy_score
import wandb

def train(
    model,
    num_epochs,
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
        for batch_x, batch_y in tqdm(training_generator, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            loss = one_epoch(model, batch_x, batch_y, optimiser, loss_fn, GPU)

        if calculate_accuracy:
            print("Calculating Accuracy")

            acc_t = train_accuracy(model, training_generator, GPU, 100)
            train_acc.append(acc_t)
            print(f"Train Accuracy: {acc_t:.2f}%")

            acc_v = train_accuracy(model, validation_generator, GPU, 100)
            test_acc.append(acc_v)
            print(f"Test Accuracy: {acc_v:.2f}%")

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


def one_epoch(
    model,
    batch_x,
    batch_y,
    optimiser,
    loss_fn,
    GPU = torch.device("cuda"),
):
    batch_x, batch_y = batch_x.to(GPU), batch_y.long().flatten().to(GPU)
    optimiser.zero_grad()
    output = model(batch_x)
    loss = loss_fn(output, batch_y)
    loss.backward()
    optimiser.step()
    
    return loss


def train_accuracy(
    model,
    data_generator,
    GPU = torch.device("cuda"),
    n = np.inf
):
    accs = []
    j = 0
    for batch_x, batch_y in data_generator:
        batch_x, batch_y = batch_x.to(GPU), batch_y.flatten()
        output = model(batch_x)
        acc = accuracy_score(batch_y.numpy(), output.argmax(1).cpu().numpy())
        accs.append(acc*100)
        j += 1
        if j == n: break
    return np.array(accs).mean()


#train and measure top1 and top5 train and test accuracy. trial version
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

def train_multiple_models(
        x, y,
        models,
        optimisers,
        loss_fns: list,
        GPU: torch.device = torch.device("cuda")
    ):
    losses = []
    for model, optimiser, loss_fn in zip(models, optimisers, loss_fns):
        loss = one_epoch(model, x, y, optimiser, loss_fn, GPU)
        losses.append(loss.item())
    return losses