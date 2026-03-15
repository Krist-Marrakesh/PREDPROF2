
import torch
import torch.nn as nn


class AudioClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_mels: int = 64,
        dropout_p1: float = 0.25,
        dropout_p2: float = 0.25,
        dropout_p3: float = 0.5,
        num_conv_blocks: int = 4,
    ):
        super().__init__()
        channels = [1, 32, 64, 128, 256]
        n = num_conv_blocks

        blocks = []

        blocks.append(nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_p1),
        ))

        for i in range(1, n - 1):
            blocks.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout_p2),
            ))

        blocks.append(nn.Sequential(
            nn.Conv2d(channels[n - 1], channels[n], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[n]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        ))

        self.features = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[n], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(num_classes: int, **kwargs) -> AudioClassifier:
    return AudioClassifier(num_classes=num_classes, **kwargs)
