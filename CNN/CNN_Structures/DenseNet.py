import torch
import torch.nn as nn
import torch.nn.functional as F

# Bottleneck Layer (DenseNet에서 사용)
class BottleneckLayer(nn.Module):
    def __init__(self, input_channels, growth_rate):
        super(BottleneckLayer, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, growth_rate * 4, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(growth_rate * 4)
        self.conv2 = nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(growth_rate)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

# Transition Layer (특징 맵을 줄이고 다운 샘플링)
class TransitionLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = F.avg_pool2d(x, 2, 2)  # 다운 샘플링
        return x

# DenseNet 구조
class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, num_blocks=[6, 12, 24, 16], num_classes=1000):
        super(DenseNet121, self).__init__()
        self.growth_rate = growth_rate

        # Initial Convolution and MaxPool
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn0 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 첫 번째 DenseBlock
        self.dense_block1 = self._make_dense_block(6, growth_rate, 64)
        self.transition1 = TransitionLayer(64 + 6 * growth_rate, (64 + 6 * growth_rate) // 2)

        # 두 번째 DenseBlock
        self.dense_block2 = self._make_dense_block(12, growth_rate, 64 + 6 * growth_rate // 2)
        self.transition2 = TransitionLayer(64 + 12 * growth_rate, (64 + 12 * growth_rate) // 2)

        # 세 번째 DenseBlock
        self.dense_block3 = self._make_dense_block(24, growth_rate, 64 + 12 * growth_rate // 2)
        self.transition3 = TransitionLayer(64 + 24 * growth_rate, (64 + 24 * growth_rate) // 2)

        # 네 번째 DenseBlock
        self.dense_block4 = self._make_dense_block(16, growth_rate, (64 + 24 * growth_rate) // 2)

        # Fully Connected Layer
        self.fc = nn.Linear(64 + 16 * growth_rate, num_classes)

    def _make_dense_block(self, num_layers, growth_rate, input_channels):
        layers = []
        for _ in range(num_layers):
            layers.append(BottleneckLayer(input_channels, growth_rate))
            input_channels += growth_rate  # 각 레이어마다 growth_rate만큼 채널 수 증가
        return nn.ModuleList(layers)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.pool0(x)

        # 첫 번째 DenseBlock
        for layer in self.dense_block1:
            out = layer(x)
            x = torch.cat((x, out), 1)  # Feature map을 concat으로 연결
        x = self.transition1(x)

        # 두 번째 DenseBlock
        for layer in self.dense_block2:
            out = layer(x)
            x = torch.cat((x, out), 1)  # Feature map을 concat으로 연결
        x = self.transition2(x)

        # 세 번째 DenseBlock
        for layer in self.dense_block3:
            out = layer(x)
            x = torch.cat((x, out), 1)  # Feature map을 concat으로 연결
        x = self.transition3(x)

        # 네 번째 DenseBlock
        for layer in self.dense_block4:
            out = layer(x)
            x = torch.cat((x, out), 1)  # Feature map을 concat으로 연결

        # 최종적으로 FC로 연결
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 모델 생성
model = DenseNet121(growth_rate=32, num_blocks=[6, 12, 24, 16], num_classes=1000)

"""
단계	                 출력 채널 수	출력 크기
입력 이미지	                    3     224x224
Conv0	                    64	   112x112
첫 번째 DenseBlock	         256     112x112
Transition Layer 1	       128	   56x56
두 번째 DenseBlock	         512     56x56
Transition Layer 2	       256	   28x28
세 번째 DenseBlock	        1024     28x28
Transition Layer 3	       512	   14x14
네 번째 DenseBlock	        1024     14x14
Adaptive Average Pooling   1024	    1x1
Fully Connected Layer	  	        클래스 수
"""
