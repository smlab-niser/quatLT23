import numpy as np
from utils import load_imagenet
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
# import torchvision
from resnet import ResNet152

hparams = {
    "batch_size": 256,
    "num_epochs": 40,
    "model": "resnet152 custom built",
    "dataset": "imagenet",
    "optimizer": "sgd",
    "learning_rate": 0.05,
    "gpu": 0,
}

GPU = torch.device(f'cuda:{hparams["gpu"]}')
CPU = torch.device('cpu')

log = True

model_save_name = f"B={hparams['batch_size']}_E={hparams['num_epochs']}_O={hparams['optimizer']}_ll={hparams['learning_rate']}.pth"

# model = ResNet152()
model = ResNet152()
model.to(GPU)
optimiser = torch.optim.SGD(model.parameters(), lr=hparams["learning_rate"])

if log:
    import wandb
    wandb.init(project="QuatLT23", name="ResNet152 Final Try run 2", config=hparams)
    wandb.watch(model)

print("Loading data...")
(x, y), (x_val, y_val) = load_imagenet()
m = len(x)

batch_size = hparams["batch_size"]
num_epochs = hparams["num_epochs"]

# for epoch in range(num_epochs):

#     t0 = time.time()

#     pbar = tqdm(total=m//batch_size+1, desc=f"Epoch {epoch+1}/{num_epochs}")

#     for i in range(0, len(x), batch_size):
#         batch_x, batch_y = x[i:i+batch_size].to(GPU), y[i:i+batch_size].to(GPU)
#         optimiser.zero_grad()
#         output = model(batch_x)
#         loss = F.mse_loss(output, batch_y)
#         loss.backward()
#         optimiser.step()

#         pbar.update(1)

#     t1 = time.time()

#     torch.save(model, model_save_name)

#     t2 = time.time()

#     ratio = 0.005
#     train_mask = np.random.choice(a=[False, True], size=m, p=[1-ratio, ratio])
#     val_mask   = np.random.choice(a=[False, True], size=len(y_val), p=[1-ratio, ratio])
#     model = model.to(CPU)
#     train_accuracy = accuracy_score(y[train_mask].argmax(1),     model(x[train_mask]).argmax(1))
#     test_accuracy  = accuracy_score(y_val[val_mask].argmax(1), model(x_val[val_mask]).argmax(1))
#     model = model.to(GPU)

#     t3 = time.time()

#     if log: wandb.log(
#         {
#             "train_acc": train_accuracy,
#             "test_acc": test_accuracy,
#             "loss": loss.item(),
#             "training_time": t1-t0,
#             "saving_time": t2-t1,
#             "accuracy_time": t3-t2,
#         }
#     )

#     pbar.close()

train_acc, test_acc = [], []

for epoch in range(num_epochs):
    print("Training")
    for i in trange(0, len(x), batch_size):
        batch_x, batch_y = x[i:i+batch_size].to(GPU), y[i:i+batch_size].float().to(GPU)
        optimiser.zero_grad()
        output = model(batch_x)
        # print(f"{output.shape = }, {batch_y.shape = }")
        loss = F.mse_loss(output, batch_y)
        loss.backward()
        optimiser.step()
    # losses.append(loss.item())
    
    print("Calculating Accuracy")
    
    train_accs, test_accs = [], []
    batch_size_acc = 450

    # for i in range(0, len(x)/10, batch_size_acc):
    for i in trange(0, 45000, batch_size_acc):
        batch_x, batch_y = x[i:i+batch_size_acc].to(GPU), y[i:i+batch_size_acc].numpy()
        train_pred = model(batch_x)
        acc = accuracy_score(batch_y.argmax(1), train_pred.argmax(1).cpu().numpy())
        train_accs.append(acc*100)
    train_acc.append(np.array(train_accs).mean())
    print(f"Train Accuracy: {train_acc[-1]:.2f}%")

    # for i in range(0, len(x_val), batch_size_acc):
    for i in trange(0, 45000, batch_size_acc):
        batch_x_test, batch_y_test = x_val[i:i+batch_size_acc].to(GPU), y_val[i:i+batch_size_acc].numpy()
        test_pred = model(batch_x_test)
        acc = accuracy_score(batch_y_test.argmax(1), test_pred.argmax(1).cpu().numpy())
        test_accs.append(acc*100)
    test_acc.append(np.array(test_accs).mean())
    print(f"Test Accuracy: {test_acc[-1]:.2f}%")

    if log: wandb.log(
        {
            "train_acc": train_acc[-1],
            "test_acc": test_acc[-1],
            "loss": loss.item(),
        }
    )
    
    torch.save(model, model_save_name)

wandb.finish()
