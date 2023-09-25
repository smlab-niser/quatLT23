import torch
import torch.nn as nn
from utils.pruning import prune_model, reset_model
from htorch import layers as quatnn


# nn.Conv2d            => quatnn.QConv2d
# nn.Linear            => quatnn.QLinear


""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""


architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],

    # next 4 layers not used for pretraining
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = quatnn.QConv2d(in_channels//4, out_channels//4, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.leakyrelu(x)
    
def create_conv_layers(in_channels, architecture):
    layers = []
    in_channels = in_channels

    for x in architecture:
        if type(x) == tuple:
            layers += [
                CNNBlock(
                    in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                )
            ]
            in_channels = x[1]

        elif type(x) == str:
            layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        elif type(x) == list:
            conv1 = x[0]
            conv2 = x[1]
            num_repeats = x[2]

            for _ in range(num_repeats):
                layers += [
                    CNNBlock(
                        in_channels,
                        conv1[1],
                        kernel_size=conv1[0],
                        stride=conv1[2],
                        padding=conv1[3],
                    )
                ]
                layers += [
                    CNNBlock(
                        conv1[1],
                        conv2[1],
                        kernel_size=conv2[0],
                        stride=conv2[2],
                        padding=conv2[3],
                    )
                ]
                in_channels = conv2[1]

    return nn.Sequential(*layers)

class PretrainingBase(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.architecture = architecture_config[:-4]
        self.in_channels = in_channels
        self.conv_layers = create_conv_layers(self.in_channels, self.architecture)

    def forward(self, x):
        return self.conv_layers(x)

class Pretraining(nn.Module):
    def __init__(self, in_channels=3, S=7, name="pretraining_quat"):
        super().__init__()
        self.pret_base = PretrainingBase(in_channels)
        self.fcs = self._create_fcs(S)
        self.name = name
    def __str__(self):
        return self.name

    def forward(self, x):
        x = self.pret_base(x)
        return self.fcs(x)

    def _create_fcs(self, S):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 1000),
        )


class YoloBase(nn.Module):
    def __init__(self, in_channels=1024):
        super().__init__()
        self.architecture = architecture_config[-4:]
        self.in_channels = in_channels
        self.yolo_base = create_conv_layers(self.in_channels, self.architecture)

    def forward(self, x):
        return self.yolo_base(x)

class Yolov1(nn.Module):
    def __init__(self, base_path, in_channels=3, S=7, B=2, C=20, name="yolov1_quat", device=torch.device("cpu")):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.base_path = base_path
        self.in_channels = in_channels
        self.pret_base = PretrainingBase(in_channels)
        self.yolo_base = YoloBase(in_channels=1024)
        self.fcs = self._create_fcs(S, B, C)
        self.name = name
        self.reset(device)
    def __str__(self):
        return self.name

    def forward(self, x):
        x = self.pret_base(x)
        x = self.yolo_base(x)
        return self.fcs(x)

    def _create_fcs(self, S, B, C):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            quatnn.QLinear(256 * S * S, 1024),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)),
        )

    def reset(self, device, seed=21):
        torch.manual_seed(seed)
        self.apply(reset_model)
        pret_base = torch.load(self.base_path, map_location=device)
        self.pret_base = pret_base.pret_base


if __name__ == "__main__":
    test = "yolo"
    if test == "yolo":
        x = torch.randn((10, 3, 448, 448))
        model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
    else:
        x = torch.randn((10, 3, 448, 448))
        model = Pretraining(split_size=7, num_boxes=2, num_classes=20)
    print(x.shape)
    y_pred = model(x)
    print(y_pred.shape)