import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """
    A ResNet block.
    Code from open_lth repository.
    Copyright (c) Facebook, Inc. and its affiliates.
    """
    def __init__(self, in_channels: int, out_channels: int, downsample=False):
        super().__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """
    Code from open_lth repository.
    Copyright (c) Facebook, Inc. and its affiliates.
    """
    def __init__(self):
        super().__init__()
        num_segments = 3
        filters_per_segment = [16, 32, 64]
        architecture = [(num_filters, num_segments) for num_filters in filters_per_segment]

        # Initial convolutional layer.
        current_filters = architecture[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # ResNet blocks
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(architecture):
            for block_index in range(num_blocks):          
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer.
        self.fc = nn.Linear(architecture[-1][0], 1000)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.blocks(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x