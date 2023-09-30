import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        self.e11 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.e12 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.e22 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.e32 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.e42 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.e52 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d11 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.d12 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d21 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.d22 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d31 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.d32 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )

        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d41 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )
        self.d42 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # BatchNorm2d added here
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.outconv = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = self.e11(x)
        xe12 = self.e12(xe11)
        xp1 = self.pool1(xe12)

        xe21 = self.e21(xp1)
        xe22 = self.e22(xe21)
        xp2 = self.pool2(xe22)

        xe31 = self.e31(xp2)
        xe32 = self.e32(xe31)
        xp3 = self.pool3(xe32)

        xe41 = self.e41(xp3)
        xe42 = self.e42(xe41)
        xp4 = self.pool4(xe42)

        xe51 = self.e51(xp4)
        xe52 = self.e52(xe51)

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.d11(xu11)
        xd12 = self.d12(xd11)

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.d21(xu22)
        xd22 = self.d22(xd21)

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.d31(xu33)
        xd32 = self.d32(xd31)

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.d41(xu44)
        xd42 = self.d42(xd41)

        # Output layer
        out = torch.sigmoid(self.outconv(xd42))

        return out