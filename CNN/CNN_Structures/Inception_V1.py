import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()

        # 1x1 Convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 1),
            nn.ReLU()
        )

        # 1x1 Convolution -> 3x3 Convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.ReLU()
        )

        # 1x1 Convolution -> 5x5 Convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 5, padding = 2),
            nn.ReLU()
        )
        
        # 3x3 MaxPool -> 1x1 convolution
        self.maxpool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            nn.Conv2d(in_channels, 32, kernel_size = 1)
        )

    def forward(self, x):
        # 따로따로 연산함
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.maxpool1(x)

        # 합치기
        return torch.cat([x1, x2, x3, x4], dim=1) 
