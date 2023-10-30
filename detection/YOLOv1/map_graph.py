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
B = 1024
DEVICE = torch.device('cuda:0')
base_dir = "/home/aritra/project/quatLT23/detection/YOLOv1/saved_models/"

training_generator = DataLoader(PascalVOC("train"), batch_size=B, shuffle=True, num_workers=8, pin_memory=True)
val_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    lambda img: torch.cat((img, F.rgb_to_grayscale(img)), dim=0),
])
validation_generator = DataLoader(PascalVOC("val", transform=val_transform), batch_size=B, shuffle=True, num_workers=8, pin_memory=True)

model_name = "real/120"

model = torch.load(f"{base_dir}/train_{model_name}.pt", map_location=DEVICE)

train_maps = {}
val_maps = {}
for th in tqdm(np.arange(0, 1, 0.05)):
    train_maps[th] = mAP(model, training_generator, DEVICE, iou_threshold_nms=th, threshold_nms=th, iou_threshold_map=th)
    val_maps[th] = mAP(model, validation_generator, DEVICE, iou_threshold_nms=th, threshold_nms=th, iou_threshold_map=th)

with open(f"{base_dir}/train_{model_name}_train_map.json", "w") as f: dump(train_maps, f)
with open(f"{base_dir}/train_{model_name}_val_map.json", "w") as f: dump(val_maps, f)