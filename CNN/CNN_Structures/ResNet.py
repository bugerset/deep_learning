import torch
import torch.nn as nn

# Bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample = None, stride = 1):
        super(Bottleneck, self).__init__()

        # Conv1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        # Conv2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=  3, stride = 1, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        # Conv3
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size = 1, stride = 1, padding = 0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()

        self.stride = stride
        self.i_downsample = i_downsample

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)

        return x
    
class ResNet(nn.Module):
    def __init__(self, Resblock, layer_list, num_classes, num_channels = 3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(Resblock, layer_list[0], planes = 64, stride = 1)
        self.layer2 = self._make_layer(Resblock, layer_list[1], planes = 128, stride = 2)
        self.layer3 = self._make_layer(Resblock, layer_list[2], planes = 256, stride = 2)
        self.layer4 = self._make_layer(Resblock, layer_list[3], planes = 512, stride = 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def _make_layer(self, Resblock, blocks, planes, stride = 1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * Resblock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * Resblock.expansion, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(planes * Resblock.expansion)
            )

        layers.append(Resblock(self.in_channels, planes, i_downsample = ii_downsample, stride = stride))

        self.in_channels = planes * Resblock.expansion

        for i in range(blocks - 1):
            layers.append(Resblock(self.in_channels, planes))

        return nn.Sequential(*layers)

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)

def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels)
