import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
import wandb

class Imagenet_Train(torch.utils.data.Dataset):
    def __init__(self, length = 1281166, base_dir="/home/aritra/project/quartLT23/data/ILSVRC", d4 = True):
        self.base_dir = base_dir+"/train_npy"
        self.length = length
        self.d4 = d4
        self.mat = np.array(
            [
                [1, 0, 0, 0.299],
                [0, 1, 0, 0.587],
                [0, 0, 1, 0.144]
            ]
        )
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        data = np.load(f"{self.base_dir}/img_{index}.npy")
        # print(data.shape)
        if self.d4:
            b = data[1:].reshape(224,224,3)
            x = np.dot(b, self.mat).transpose(2, 0, 1)
        else: x = data[1:].reshape(224, 224, 3).transpose(2,0,1)
        return torch.from_numpy(x).float(), torch.tensor([data[0]]).float()
   
class Imagenet_Val(torch.utils.data.Dataset):
    def __init__(self, length = 50000, base_dir="/home/aritra/project/quartLT23/data/ILSVRC", d4 = True):
        self.base_dir = base_dir+"/test_npy"
        self.length = length
        self.d4 = d4
        self.mat = np.array(
            [
                [1, 0, 0, 0.299],
                [0, 1, 0, 0.587],
                [0, 0, 1, 0.144]
            ]
        )
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        data = np.load(f"{self.base_dir}/img_{index}.npy")
        # print(data.shape)
        if self.d4:
            b = data[1:].reshape(224,224,3)
            x = np.dot(b, self.mat).transpose(2, 0, 1)
        else: x = data[1:].reshape(224, 224, 3).transpose(2,0,1)
        return torch.from_numpy(x).float(), torch.tensor([data[0]]).float()
      
def load_imagenet(n=10):
    base_dir="/home/aritra/project/quartLT23/data/ILSVRC"

    train_files = [f"{base_dir}/train_npy/img_{i}.npy" for i in range(n)]

    # x_train, y_train = None, None
    train = None

    for i in range(n):
        if train is None:
            train = np.array([np.load(train_files[i])])
        else:
            train = np.append(train, [np.load(train_files[i])], axis=0)

    test_files = [f"{base_dir}/test_npy/img_{i}.npy" for i in range(n)]

    # x_train, y_train = None, None
    test = None

    for i in range(n):
        if test is None:
            test = np.array([np.load(test_files[i])])
        else:
            test = np.append(test, [np.load(test_files[i])], axis=0)
            
            
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
   
   
class Imagenet64_train(torch.utils.data.Dataset):
    def __init__(self, length = 1281167, base_dir="/mnt/data/datasets/imagenet/64x64", d4 = True):
        self.base_dir = base_dir+"/train"
        self.length = length
        self.d4 = d4
        self.mat = np.array(
            [
                [1, 0, 0, 0.299],
                [0, 1, 0, 0.587],
                [0, 0, 1, 0.144]
            ]
        )
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        data = np.load(f"{self.base_dir}/{index}.npy")
        if self.d4:
            b = data[1:].reshape(3, 64, 64).transpose(1, 2, 0)
            x = np.dot(b, self.mat).transpose(2, 0, 1)
        else: x = data[1:].reshape(3, 64, 64)
        return torch.from_numpy(x).float(), torch.tensor([data[0]-1]).float()
        

class Imagenet64_val(torch.utils.data.Dataset):
    def __init__(self, length = 50000, base_dir="/mnt/data/datasets/imagenet/64x64", d4 = True):
        self.base_dir = base_dir+"/data/val"
        self.length = length
        data = np.load(f"{self.base_dir}/val_data", allow_pickle=True)
        self.x = torch.from_numpy((data["data"]/255).reshape(-1, 3, 64, 64)).float()
        self.y = torch.from_numpy(np.array(data["labels"])-1).float()
        if d4: self.x = make4D(self.x)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def one_hot(y):
    # y = y.numpy()
    len_y = len(y)
    ret = np.zeros((len_y, 1000), dtype=bool)
    ret[np.arange(len_y), y-1] = True
    return ret.astype(bool)

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

def make4D(x):
    """Takes an array of size (n, 3, 64, 64) and returns an array of size (n, 4, 64, 64). Adds an extra leyer to each image where the last layer is a grayscale version of the image.

    Args:
        x (np.ndarray): Array of size (n, 3, 64, 64)
        
    Returns:
        np.ndarray: Array of size (n, 4, 64, 64)
    """

    mat = torch.tensor(
        [
            [1, 0, 0, 0.299],
            [0, 1, 0, 0.587],
            [0, 0, 1, 0.144]
        ]
    ).float().to(x.device)  # RGB to RGB+Grayscale conversion matrix
    x = x.permute(0, 2, 3, 1)
    x = torch.matmul(x, mat)
    return x.permute(0, 3, 1, 2)


# if __name__ == "__main__":
#     x = torch.from_numpy(np.random.rand(10, 3, 64, 64)).float().cuda()
#     # (x, y), (x_val, y_val) = load_imagenet(1)
#     # b = make4D(x[:5000])
#     # print(f"{x.shape} -> {b.shape}")
#     print(x.device)
if __name__ == "__main__":
    l = Imagenet_Train()
    print(l[0])
    
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