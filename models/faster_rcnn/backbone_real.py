import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        
        # # in a more explainable way
        # res = self.conv1(x)
        # skipped = self.shortcut(x)

        # res = self.bn1(res)
        # res = F.relu(res)
        # res = self.conv2(res)
        # res = self.bn2(res)

        # out = res+skipped
        
        # out = F.relu(out)
        # return out


        # # in a more concise way
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, img_channel=4, name = "ResNet_real"):
        super(ResNet, self).__init__()
        self.name = name
        self.in_planes = 64
        self.conv1 = nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def __repr__(self):
        return self.name


def ResNet18(img_channel=4, num_classes=1000, name = "ResNet18"):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, img_channel, name=name)

def ResNet34(img_channel=4, num_classes=1000, name = "ResNet34"):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, img_channel, name=name)

def ResNet50(img_channel=4, num_classes=1000, name = "ResNet50"):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, img_channel, name=name)

def ResNet101(img_channel=4, num_classes=1000, name = "ResNet101"):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, img_channel, name=name)

def ResNet152(img_channel=4, num_classes=1000, name = "ResNet152"):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, img_channel, name=name)


if __name__ == "__main__":
    net = ResNet152().cuda(1)
    batch_x = torch.randn(256, 4, 32, 32).cuda(1)
    print(batch_x.size(), end = " -> ", flush=True)
    y = net(batch_x)
    print(y.size())


# RN18, RN34
# [256, 4, 224, 224] -> [256, 512, 28, 28]
# [256, 4,  64,  64] -> [256, 512,  8,  8]
# [256, 4,  32,  32] -> [256, 512,  4,  4]

# RN50, RN101, RN152
# [256, 4, 224, 224] -> [256, 2048,  28,  28]
# [256, 4,  64,  64] -> [256, 2048,   8,   8]
# [256, 4,  32,  32] -> [256, 2048,   4,   4]