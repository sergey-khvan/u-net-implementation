import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.conv1 = Block(in_channels=in_channels, out_channels=64)
        self.conv2 = Block(in_channels=64, out_channels=128)
        self.conv3 = Block(in_channels=128, out_channels=256)
        self.conv4 = Block(in_channels=256, out_channels=512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bot_block = Block(in_channels=512, out_channels=1024)

        self.exp1 = Block(in_channels=128, out_channels=64)
        self.exp2 = Block(in_channels=256, out_channels=128)
        self.exp3 = Block(in_channels=512, out_channels=256)
        self.exp4 = Block(in_channels=1024, out_channels=512)
        self.upconv_b = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.maxpool(x1)
        x2 = self.conv2(x)
        x = self.maxpool(x2)
        x3 = self.conv3(x)
        x = self.maxpool(x3)
        x4 = self.conv4(x)
        x = self.maxpool(x4)
        x = self.bot_block(x)
        x = self.upconv_b(x)
        x = torch.cat((x4, x), dim=1)
        x = self.exp4(x)
        x = self.upconv4(x)
        x = torch.cat((x3, x), dim=1)
        x = self.exp3(x)
        x = self.upconv3(x)
        x = torch.cat((x2, x), dim=1)
        x = self.exp2(x)
        x = self.upconv2(x)
        x = torch.cat((x1, x), dim=1)
        x = self.exp1(x)
        x = self.outconv(x)
        return x
