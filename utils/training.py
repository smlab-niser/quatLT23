import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import wandb

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