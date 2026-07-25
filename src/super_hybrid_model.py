import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ChannelAttention(nn.Module):
    """Channel attention for CBAM."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attn = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial attention for CBAM."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(combined))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-Excitation block for channel recalibration."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.cbam = CBAM(channels=2048, reduction=16)
        self.se = SEBlock(channels=2048, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.se(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


class DeepHybridCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.cbam = CBAM(channels=512, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.cnn_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.hc_fc = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

        self.final_fc = nn.Sequential(
            nn.Linear(512 + 128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def _handcrafted_stats(self, x):
        eps = 1e-6
        mean = x.mean(dim=[2, 3])
        std = x.std(dim=[2, 3]) + eps
        centered = x - mean[:, :, None, None]
        skew = (centered.pow(3).mean(dim=[2, 3]) / (std.pow(3) + eps))
        kurt = (centered.pow(4).mean(dim=[2, 3]) / (std.pow(4) + eps))
        return torch.cat([mean, std, skew, kurt], dim=1)

    def forward(self, x):
        b = x.size(0)

        cnn = self.conv_layers(x)
        cnn = self.cbam(cnn)
        cnn = self.pool(cnn)
        cnn = cnn.view(b, -1)
        cnn = self.cnn_fc(cnn)

        hc = self._handcrafted_stats(x)
        hc = self.hc_fc(hc)

        combined = torch.cat([cnn, hc], dim=1)
        out = self.final_fc(combined)
        return out


class SuperHybridModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.branch1 = ResNet50Backbone()
        self.branch2 = DeepHybridCNN()
        self.classifier = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        feats = torch.cat([f1, f2], dim=1)
        out = self.classifier(feats)
        return out, feats

