import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)

        self.relu = nn.ReLU(inplace = True)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride = stride)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        shortcut = self.shortcut(x)

        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, Depth, Width, num_classes = 1000):
        super(WideResNet, self).__init__()

        k = Width
        l = int((Depth - 4) / 6)

        self.in_channels = 16
        self.init_conv = nn.Conv2d(3, self.in_channels, 3, 1, padding=1)

        self.conv2 = self._make_layer(16 * k, l, 1)
        self.conv3 = self._make_layer(32 * k, l, 2)
        self.conv4 = self._make_layer(64 * k, l, 2)

        self.bn = nn.BatchNorm2d(64 * k)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * k, num_classes)
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def _make_layer(self, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for stride in strides:
            layers.append(ResBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
    
def WRN_22_8():
    net = WideResNet(22, 8, num_classes = 10)
    return net

def WRN_40_4():
    net = WideResNet(40, 4, num_classes = 10)
    return net

def WRN_28_10():
    net = WideResNet(28, 10, num_classes = 10)
    return net
