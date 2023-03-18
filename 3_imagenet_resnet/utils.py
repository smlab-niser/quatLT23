from os import listdir
import numpy as np

def s(x):
    if x[-2] == "_": return int(x[-1])
    if x[-3] == "_": return int(x[-2:])

def load_imagenet(n = 10):
    base_dir = "/mnt/data/datasets/imagenet/64x64"

    train_files = [
        f"{base_dir}/train/{name}"
            for name in listdir(base_dir+"/train")
            if not name.endswith(".zip")
    ]
    
    train_files = sorted(train_files, key=s)[0:n]
    print(*train_files[51:], sep="\n")

    parts = list(map(
        lambda x : np.load(x, allow_pickle=True),
        train_files
    ))

    x_train = np.concatenate([part["data"]   for part in parts], axis=0)/255
    y_train = np.concatenate([part["labels"] for part in parts], axis=0)

    val_file = base_dir + "/val/val_data"
    val = np.load(val_file, allow_pickle=True)

    x_val = val["data"]/255
    y_val = np.array(val["labels"])

    return (x_train, y_train), (x_val, y_val)