print("using conv_6 model on cifar10 dataset")

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
# import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from htorch import layers
import numpy as np
import json
import datetime

print(datetime.datetime.now())
dataset = "cifar10"

print(f"Loading {dataset} dataset")
device = torch.device('cuda:0')

data = pd.read_csv(f"../data/{dataset}/train.csv", header=None).to_numpy()
x, y = (data[:, 1:]/255).reshape(-1, 3, 32, 32).transpose((0, 3, 2, 1)), torch.nn.functional.one_hot(torch.Tensor(data[:, 0]).long(), 10).to(device)

test = pd.read_csv(f"../data/{dataset}/test.csv", header=None).to_numpy()
x_test, y_test = (test[:, 1:]/255).reshape(-1, 3, 32, 32).transpose((0, 3, 2, 1)), torch.nn.functional.one_hot(torch.Tensor(test[:, 0]).long(), 10).to(device)

good_boy = np.array(
    [[1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0.299, 0.587, 0.144]]
).T

x = torch.Tensor(np.dot(x, good_boy).transpose((0, 3, 1, 2))).float().to(device)
x_test = torch.Tensor(np.dot(x_test, good_boy).transpose((0, 3, 1, 2))).float().to(device)

class Real(nn.Module):
    def __init__(self, out_channels: int = 10):
        super().__init__()
        self.conv11 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out_channels)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool(x)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.pool(x)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Quat(nn.Module):
    def __init__(self, out_channels: int = 10):
        super().__init__()
        self.conv11 = layers.QConv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv12 = layers.QConv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv21 = layers.QConv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv22 = layers.QConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv31 = layers.QConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv32 = layers.QConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = layers.QLinear(64 * 4 * 4, 64)
        self.fc2 = layers.QLinear(64, 64)
        self.fc3 = nn.Linear(256, out_channels)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool(x)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.pool(x)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


batchsizes = []
real_accu = [[],[]] #train and test
quat_accu = [[],[]]
real_time = [[],[]] #forward and backward
quat_time = [[],[]]

epochs = 80

conf = 3

for batch_size in tqdm([i for i in range(100, 8000, 100)][::-1]):
    batchsizes.append(batch_size)
    for model in ["real", "quat"]:
        if model == "quat":
            model = Quat()
            model.to(device)
        elif model == "real":
            model = Real()
            model.to(device)

        optimiser = torch.optim.Adam(model.parameters(), lr=1.2e-3)
        
        # print(f"batchsize: {batch_size} model: {model.__class__.__name__}")
        
        t_f = 0
        t_b = 0

        train_accuracy = 0
        test_accuracy = 0
        
        for i in range(conf):
            for epoch in range(epochs):
                for i in range(0, len(x), batch_size):
                    batch_x, batch_y = x[i:i+batch_size], y[i:i+batch_size].float()
                    optimiser.zero_grad()
                    
                    t0 = time.time()
                    output = model(batch_x)
                    t_f += time.time() - t0
                    
                    t0 = time.time()
                    loss = F.mse_loss(output, batch_y)
                    loss.backward()
                    t_b += time.time() - t0

                    optimiser.step()
                
            y_pred = torch.cat([model(c).detach().cpu() for c in torch.chunk(x, 4)])
            y_pred_test = model(x_test).detach().cpu()
            
            train_accuracy += accuracy_score(y.argmax(1).cpu(), y_pred.argmax(1))*100
            test_accuracy += accuracy_score(y_test.argmax(1).cpu(), y_pred_test.argmax(1))*100

        # print(f"train accuracy: {train_accuracy} test accuracy: {test_accuracy}")
        # print(f"forward time: {t_f} backward time: {t_b}")
        
        if isinstance(model, Real):
            real_accu[0].append(train_accuracy/conf)
            real_accu[1].append(test_accuracy/conf)
            real_time[0].append(t_f*1000/(epochs*conf))
            real_time[1].append(t_b*1000/(epochs*conf))
        else:
            quat_accu[0].append(train_accuracy/conf)
            quat_accu[1].append(test_accuracy/conf)
            quat_time[0].append(t_f*1000/(epochs*conf))
            quat_time[1].append(t_b*1000/(epochs*conf))

results = {
    "batchsizes": batchsizes,
    "real_accu": real_accu,
    "quat_accu": quat_accu,
    "real_time": real_time,
    "quat_time": quat_time
}

print(results)

with open("../docs/results3.json", "w") as f:
    json.dump(results, f)

print(datetime.datetime.now())