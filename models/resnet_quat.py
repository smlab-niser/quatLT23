# nn.Conv2d            => quatnn.QConv2d
# nn.BatchNorm2D       => quatnn.QBatchNorm2d
# nn.MaxPool2d         => quatnn.QMaxPool2d
# nn.AdaptiveAvgPool2d 
# nn.Linear            => quatnn.QLinear
# nn.ReLU()
# nn.Sequential


import torch.nn as nn
from htorch import layers as quatnn

class Block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        # print(f"Block: {in_channels = } {intermediate_channels = } {stride = }")
        super().__init__()
        self.expansion = 4
        self.conv1  = quatnn.QConv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1    = quatnn.QBatchNorm2d(intermediate_channels)
        self.conv2  = quatnn.QConv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2    = quatnn.QBatchNorm2d(intermediate_channels)
        self.conv3  = quatnn.QConv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3    = quatnn.QBatchNorm2d(intermediate_channels * self.expansion)
        self.relu   = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        # print(f"\tBlock: {x.shape = }")
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # print(f"\t\tBefore going into bn2: {x.shape = }")
        x = self.bn2(x)
        x = self.relu(x)
        # print(f"\tBlock: {x.shape = }")
        x = self.conv3(x)
        # try:
        #     x = self.conv3(x)
        # except Exception as e:
        #     print(f"Exception: {x.shape = } = {e}")
        #     raise e
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64//4
        self.conv1 = quatnn.QConv2d(
            image_channels//4, 64//4, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = quatnn.QBatchNorm2d(64//4)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64//4, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128//4, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256//4, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512//4, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = quatnn.QLinear(512, num_classes//4)

    def forward(self, x):
        # print(f"Before conv1: {x.shape = }")
        x = self.conv1(x)
        # print(f"After conv1: {x.shape = }")
        x = self.bn1(x)
        # print(f"After bn1: {x.shape = }")
        x = self.relu(x)
        # print(f"After relu: {x.shape = }")
        x = self.maxpool(x)
        # print(f"After maxpool: {x.shape = }\n")

        x = self.layer1(x)  #  64, 1
        # print(f"After layer1: {x.shape = }")
        x = self.layer2(x)  # 128, 2
        # print(f"After layer2: {x.shape = }")
        x = self.layer3(x)  # 256, 2
        # print(f"After layer3: {x.shape = }")
        x = self.layer4(x)  # 512, 2
        # print(f"After layer4: {x.shape = }\n")

        x = self.avgpool(x)
        # print(f"After avgpool: {x.shape = }")
        x = x.reshape(x.shape[0], -1)
        # print(f"After reshaping: {x.shape = }")
        x = self.fc(x)
        # print(f"After fc: {x.shape = }")

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                quatnn.QConv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                quatnn.QBatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50, 101, 152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

def ResNet18(img_channel=3, num_classes=1000):
    return ResNet(Block, [2, 2, 2, 2], img_channel, num_classes)

def ResNet34(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 4, 6, 3], img_channel, num_classes)

def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 4, 6, 3], img_channel, num_classes)

def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 4, 23, 3], img_channel, num_classes)

def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 8, 36, 3], img_channel, num_classes)

