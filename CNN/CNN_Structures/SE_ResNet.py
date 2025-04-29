import torch
import torch.nn as nn

class SEblock(nn.Module):
    def __init__(self, in_channels, reduction = 16):
        super(SEblock, self).__init__()
        
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_channels // reduction, in_channels, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        out = self.avg(x).view(batch, channels)
        out = self.fc(out).view(batch, channels, 1, 1)

        return out * x

    
class SEbasicblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(SEbasicblock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, 
                               padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1,
                               padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEblock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += self.shortcut(x)
        out = nn.ReLU()(out)

        return out

class SEResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SEResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [SEbasicblock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(SEbasicblock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
