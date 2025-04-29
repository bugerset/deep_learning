import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction = 4):
        super(SEBlock, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            Swish(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch, channels, _, _ = x.shape
        squeezed = self.squeeze(x).view(batch, channels)
        out = self.fc(squeezed).view(batch, channels, 1, 1)
        return x * out
    
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvBlock, self).__init__()

        hidden_dim = in_channels * expand_ratio
        self.use_residual = in_channels == out_channels and stride == 1

        self.block = nn.Sequential(
            # 1
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish(),

            # 2
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride,
                      padding = kernel_size // 2, groups = hidden_dim, bias = False),
            nn.BatchNorm2d(hidden_dim),
            SEBlock(hidden_dim),

            # 3
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)
        
class EfficientNet(nn.Module):
    def __init__(self, phi=0, num_classes=1000):
        super(EfficientNet, self).__init__()

        # ✅ Compound Scaling 적용 (depth, width, resolution 조절)
        depth_scale = 1.2 ** phi  # 모델 깊이 조절
        width_scale = 1.1 ** phi  # 채널 수 조절
        resolution_scale = 1.15 ** phi  # 이미지 해상도 조절

        base_channels = int(32 * width_scale)  # 첫 번째 Conv 채널 수 조절
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1, bias=False),  # 초기 Conv
            nn.BatchNorm2d(base_channels),
            Swish()
        )

        # ✅ EfficientNet Blocks (Compound Scaling 적용)
        self.blocks = nn.Sequential(
            MBConvBlock(base_channels, int(16 * width_scale), kernel_size=3, stride=1, expand_ratio=1),
            MBConvBlock(int(16 * width_scale), int(24 * width_scale), kernel_size=3, stride=2, expand_ratio=6),
            MBConvBlock(int(24 * width_scale), int(40 * width_scale), kernel_size=5, stride=2, expand_ratio=6),
            MBConvBlock(int(40 * width_scale), int(80 * width_scale), kernel_size=3, stride=2, expand_ratio=6),
            MBConvBlock(int(80 * width_scale), int(112 * width_scale), kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(int(112 * width_scale), int(192 * width_scale), kernel_size=5, stride=2, expand_ratio=6),
            MBConvBlock(int(192 * width_scale), int(320 * width_scale), kernel_size=3, stride=1, expand_ratio=6),
        )

        # ✅ Final Layers (Classifier)
        final_channels = int(1280 * width_scale)
        self.head = nn.Sequential(
            nn.Conv2d(int(320 * width_scale), final_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_channels),
            Swish(),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(final_channels, num_classes)  # Fully Connected Layer
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
