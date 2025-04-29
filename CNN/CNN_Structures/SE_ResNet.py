import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_img = datasets.CIFAR10(
    root = "./data",
    train = True,
    download = True,
    transform = train_transform
)

test_img = datasets.CIFAR10(
    root = "./data",
    train = False,
    download = True,
    transform = test_transform
)

print(train_img, '\n')

train_dataloader = DataLoader(train_img, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_img, batch_size = 64, shuffle = False)

img, label = next(iter(train_dataloader))
print("Image Size => {}".format(img.size()))

figure = plt.figure(figsize = (10, 8))
rows, cols = 4, 3

for i in range(1, 1 + rows * cols):
    idx = torch.randint(len(train_img), size = (1,)).item()
    img, label = train_img[idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    undo_img = img * 0.5 + 0.5
    plt.imshow(to_pil_image(undo_img))
    
plt.show()

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

import torchsummary
EPOCHS = 20
LR = 1e-3 
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device on => {}".format(device))
model = SEResNet(10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR )

print(torchsummary.summary(model, (3, 224, 224), 64))

def train(train_dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(train_dataloader.dataset)

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(test_dataloader, model, loss_fn):
    model.eval()
    size = len(test_dataloader.dataset)
    batch_size = len(test_dataloader)

    loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            loss += loss_fn(pred, y)

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    loss /= batch_size

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f}")

for i in range(EPOCHS):
    print("Epoch => ({})".format(i+1))
    train(train_dataloader, model ,loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!!!")
