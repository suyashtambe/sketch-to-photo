# Generator - U-Net
from torch import nn, optim
import os
import torch

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.down = nn.Sequential(
            self.contract(in_channels, features),
            self.contract(features, features * 2),
            self.contract(features * 2, features * 4),
            self.contract(features * 4, features * 8),
            self.contract(features * 8, features * 8),
        )
        self.up = nn.Sequential(
            self.expand(features * 8, features * 8),
            self.expand(features * 16, features * 4),
            self.expand(features * 8, features * 2),
            self.expand(features * 4, features),
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def contract(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )

    def expand(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        skips = []
        for layer in self.down:
            x = layer(x)
            skips.append(x)
        skips = skips[:-1][::-1]  # reverse except bottleneck
        for i, layer in enumerate(self.up):
            x = layer(x)
            if i < len(skips):
                x = torch.cat([x, skips[i]], dim=1)
        return self.final(x)

# PatchGAN Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 4, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))
