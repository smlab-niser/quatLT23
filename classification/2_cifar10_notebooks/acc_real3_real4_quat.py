print("using conv_6 model on cifar10 dataset")

import torch, json, datetime
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import trange
from sklearn.metrics import accuracy_score
from conv_4_models import Real3, Real4, Quat

print(datetime.datetime.now())

device = torch.device('cuda:0')

dataset = "cifar10"
print(f"Loading {dataset} dataset", end="... ")
data = pd.read_csv(f"../data/{dataset}/train.csv", header=None).to_numpy()
x, y = (data[:, 1:]/255).reshape(-1, 3, 32, 32).transpose((0, 3, 2, 1)), torch.nn.functional.one_hot(torch.Tensor(data[:, 0]).long(), 10).to(device)

test = pd.read_csv(f"../data/{dataset}/test.csv", header=None).to_numpy()
x_test, y_test = (test[:, 1:]/255).reshape(-1, 3, 32, 32).transpose((0, 3, 2, 1)), torch.nn.functional.one_hot(torch.Tensor(test[:, 0]).long(), 10).to(device)

mat = np.array(
    [
        [1, 0, 0, 0.299],
        [0, 1, 0, 0.587],
        [0, 0, 1, 0.144]
    ]
)  # RGB to RGB+Grayscale conversion matrix

x_4 = torch.Tensor(np.dot(x, mat).transpose((0, 3, 1, 2))).float().to(device)
x_test_4 = torch.Tensor(np.dot(x_test, mat).transpose((0, 3, 1, 2))).float().to(device)

x_3 = torch.Tensor(x.transpose(0, 3, 1, 2)).float().to(device)
x_test_3 = torch.Tensor(x_test.transpose(0, 3, 1, 2)).float().to(device)
del x, x_test

print("done")

all_accuracies = [] # real3, real4, quat (train, test)

max_epochs = 80
# Bs = [32, 64, 128, 256, 512, 1024, 2048]  # batch size
Bs = [60]  # batch size
conf = 3  # number of times to run the same model with different initial weights to get the average accuracy

for B in Bs:
    for model_name in ["Real3", "Real4", "Quat"]:
        train_accuracies = []
        test_accuracies  = []
        
        print(f"Running model {model_name} with batch size {B}")
        
        for j in trange(conf):
            # print(f"Running model {model_name} for the {j+1}th time")
            
            if model_name == "Real3": model = Real3()
            elif model_name == "Real4": model = Real4()
            elif model_name == "Quat":  model = Quat()
            else: raise ValueError("Invalid model name")
            
            model.to(device)

            optimiser = torch.optim.Adam(model.parameters(), lr=1.2e-3)
            
            x = x_3 if isinstance(model, Real3) else x_4
            x_test = x_test_3 if isinstance(model, Real3) else x_test_4

            train_accuracy = []
            test_accuracy  = []
            for epoch in range(max_epochs):
                for i in range(0, len(x), B):
                    batch_x, batch_y = x[i:i+B], y[i:i+B].float()
                    optimiser.zero_grad()
                    output = model(batch_x)
                    loss = F.mse_loss(output, batch_y)
                    loss.backward()
                    optimiser.step()

                y_pred = torch.cat([model(c).detach().cpu() for c in torch.chunk(x, 10)])
                y_pred_test = model(x_test).detach().cpu()
                train_accuracy.append(accuracy_score(y.argmax(1).cpu(), y_pred.argmax(1))*100)
                test_accuracy.append(accuracy_score(y_test.argmax(1).cpu(), y_pred_test.argmax(1))*100)
                
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
        
        train_accuracies = np.array(train_accuracies).mean(axis=0).tolist()
        test_accuracies  = np.array(test_accuracies).mean(axis=0).tolist()
        # train and test accuracies are np array of shape (max_epochs,)
        
        all_accuracies.append((train_accuracies, test_accuracies))


    results = {
        "epochs": [i for i in range(max_epochs)],
        "real3_accu": all_accuracies[0],
        "real4_accu": all_accuracies[1],
        "quat_accu":  all_accuracies[2],
    }

    print(f"\n\n{results = }\n\n")

    with open(f"../docs/acc_real3_real4_quat_B={B}.json", "w") as f:
        json.dump(results, f)

print(datetime.datetime.now())