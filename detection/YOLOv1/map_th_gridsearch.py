import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from data_loaders.pascal_voc import PascalVOC
from models.yolo_real import Yolov1 as Real
from models.yolo_quat import Yolov1 as Quat
from utils.yolo_utils import mAP
import torchvision.transforms.functional as F
from torchvision import transforms
from json import dump
import argparse

parser = argparse.ArgumentParser(description="Gridsearch of thresholds.")
parser.add_argument("--modeltype", "-m", type=str, choices=["real", "quat"], help="Model type: real or quat")
parser.add_argument("--epochnumber", "-e", type=int, choices=[120, 140], help="Which epoch to use.")
parser.add_argument("--gpu", type=int, choices=[0, 1, 2, 3], help="Which GPU to use.")
args = parser.parse_args()

B = 1024
DEVICE = torch.device(f'cuda:{args.gpu}')
base_dir = "/home/aritra/project/quatLT23/detection/YOLOv1/saved_models/"

training_generator = DataLoader(PascalVOC("train"), batch_size=B, shuffle=True, num_workers=8, pin_memory=True)
val_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    lambda img: torch.cat((img, F.rgb_to_grayscale(img)), dim=0),
])
validation_generator = DataLoader(PascalVOC("val", transform=val_transform), batch_size=B, shuffle=True, num_workers=8, pin_memory=True)

model_name = f"{args.modeltype}/{args.epochnumber}"

model = torch.load(f"{base_dir}/train_{model_name}.pt", map_location=DEVICE)

train_maps = np.zeros((10, 10, 10))
val_maps = np.zeros((10, 10, 10))
for i, th1 in enumerate(np.arange(0, 1, 0.1)):
    for j, th2 in enumerate(tqdm(np.arange(0, 1, 0.1), desc=f"threshold_nms={th1}")):
        for k, th3 in enumerate(np.arange(0, 1, 0.1)):
            train_maps[i, j, k] = mAP(model, training_generator, DEVICE, iou_threshold_nms=th1, threshold_nms=th2, iou_threshold_map=th3)
            val_maps[i, j, k] = mAP(model, validation_generator, DEVICE, iou_threshold_nms=th1, threshold_nms=th2, iou_threshold_map=th3)

np.save(f"{base_dir}/train_{model_name}_train_map.npy", train_maps)
np.save(f"{base_dir}/train_{model_name}_val_map.npy", val_maps)

# python map_th_gridsearch.py -m real -e 120 --gpu 0