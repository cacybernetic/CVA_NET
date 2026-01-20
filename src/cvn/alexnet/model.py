from typing import List
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class Config:
    img_channels: int = 3
    img_size: int = 224
    class_names: List[str] = field(default_factory=list)
    dropout: float = 0.5


class AlexNet(nn.Module):

    def __init__(self, num_channels: int=3, img_size: int=224, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        fm_shape = self._feat_shape(num_channels, img_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fm_shape[1], 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def _feat_shape(self, num_channels: int, image_size: int) -> torch.Size:
        x = torch.zeros((1, num_channels, image_size, image_size))
        y = self.features(x)
        y = y.reshape(y.shape[0], -1)
        return y.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def init_weights_he(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        # He normal initialization for convolutional layers;
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # He normal initialization for fully connected (linear) layers;
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
