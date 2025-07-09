import torch
import torch.nn as nn

class EmotionRecognition(nn.Module):
    def __init__(self):
        super(EmotionRecognition, self).__init__()

        self.blockOne = nn.Sequential(
            nn.Conv2d(
                in_channels=3, 
                out_channels=16, 
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.blockTwo = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 12 * 12, out_features=7)
        )

    def forward(self, tnsr : torch.Tensor) -> torch.Tensor:
        tnsr = self.blockOne(tnsr)
        tnsr = self.blockTwo(tnsr)
        tnsr = self.classifier(tnsr)
        return tnsr
