import numpy as np
import torch
from tqdm import tqdm, trange

    
def one_hot(y):
    len_y = len(y)
    ret = np.zeros((len_y, 1000), dtype=bool)
    ret[np.arange(len_y), y-1] = True
    return ret

def load_imagenet(n=10):
    base_dir = "/mnt/data/datasets/imagenet/64x64"

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
    x_val = val["data"]/255
    y_val = np.array(val["labels"])
    del val
    print("val data loaded")

    x_train = torch.from_numpy(x_train.reshape(-1, 3, 64, 64)).float()
    y_train = torch.from_numpy(one_hot(y_train)).float()

    print("train data converted to tensors")

    x_val = torch.from_numpy(x_val.reshape(-1, 3, 64, 64)).float()
    y_val = torch.from_numpy(one_hot(y_val)).float()

    return (x_train, y_train), (x_val, y_val)