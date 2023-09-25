from json import load
import os
import torch
from torch import nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset#, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
# import numpy as np

def convert_annotations(annotations, S=7, C=20):
    """Converts annotations in the format (label, x, y, width, height) to a tensor of shape (S, S, C + 5).

    Args:
        annotations (list): List of annotations in the format (label, x, y, width, height).
        S (int, optional): Grid size. Defaults to 7.
        C (int, optional): Number of classes. Defaults to 20.

    Returns:
        torch.Tensor: Tensor of shape (S, S, C + 5) containing the converted annotations.
    """
    # Create empty annotation tensor
    annotation_tensor = torch.zeros((S, S, C + 5))

    for annotation in annotations:
        label, x, y, width, height = annotation

        # Calculate grid cell indices
        cell_row = int(y * S)
        cell_col = int(x * S)
        # print(cell_row, cell_col)

        # Calculate relative coordinates
        rel_x = x * S - cell_col
        rel_y = y * S - cell_row
        rel_w = width * S
        rel_h = height * S

        # Assign class label and bounding box information
        ann1 = torch.zeros(C)
        ann1[label] = 1.0
        ann2 = torch.tensor([1.0, rel_x, rel_y, rel_w, rel_h])
        annotation_tensor[cell_row, cell_col] = torch.cat((ann1, ann2))

    return annotation_tensor

class PascalVOC(Dataset):
    def __init__(self, mode, transform=None, length=None, S=7, C=20):

        with open("/home/aritra/project/quatLT23/base_dirs.json") as f: base_dir = load(f)["pascal_voc"]

        self.image_dir = base_dir+"/JPEGImages"
        self.annotation_dir = base_dir+"/yolo_annotations"
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.RandomResizedCrop(size=(448, 448), scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
            transforms.ToTensor(),
            # transforms.ToTensor() does 3 things:
            #     1. Converts PIL image to torch.Tensor
            #     2. Normalizes the pixel values between 0 and 1.
            #     3. Transposes the channel (H x W x C) to (C x H x W)
            # three2four(),
            lambda img: torch.cat((img, F.rgb_to_grayscale(img)), dim=0),
        ]) if transform is None else transform
        
        # print(f"Using {self.transform} for {mode} set")

        # self.image_list = sorted(os.listdir(self.image_dir))
        # self.annotation_list = sorted(os.listdir(self.annotation_dir))
        with open(base_dir+"/split.json") as f:
            d = load(f).get(mode)
            if d is None: raise Exception("Invalid mode!")
            self.image_list = list(map(lambda x: f"{x}.jpg", d))
            self.annotation_list = list(map(lambda x: f"{x}.txt", d))
        if len(self.image_list) != len(self.annotation_list):
            raise Exception("Number of images and annotations do not match!")

        self.length = len(self.image_list) if length is None else length
        self.S = S
        self.C = C
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = f"{self.image_dir}/{self.image_list[index]}"
        annotation_path = f"{self.annotation_dir}/{self.annotation_list[index]}"

        image = Image.open(image_path)#.convert("RGB")

        with open(annotation_path, "r") as f:
            annotation_lines = f.readlines()

        # Parse annotations
        annotations = []
        for line in annotation_lines:
            label, x_center, y_center, width, height = line.strip().split()
            label = int(label)
            x_center, y_center, width, height = map(float, (x_center, y_center, width, height))
            annotations.append((label, x_center, y_center, width, height))

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        return image, convert_annotations(annotations, self.S, self.C)


if __name__ == "__main__":

    dataset = PascalVOC("train")
    print(len(dataset))
    for image, annotation in tqdm(dataset):
        pass
        

#     dataset = PascalVOCDataset()
#     image, annotation = dataset[0]
#     print(annotation)
#     plt.imshow(image.permute(1, 2, 0))
#     plt.show()