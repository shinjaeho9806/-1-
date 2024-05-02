import torch
from torch import nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        y = self.layers(x)
        return y
    
class CNNModel(nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.out_size = out_size

        self.feature_extractor = nn.Sequential(  # (B,3,32,32)
            CNNBlock(3,32),                      # (B,32,16,16)
            CNNBlock(32,64),                     # (B,64,8,8)
            CNNBlock(64,128),                    # (B,128,4,4)
            CNNBlock(128,256),                   # (B,256,2,2)
            CNNBlock(256,512),                   # (B,512,1,1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 200),
            nn.ReLU(inplace = True),
            nn.Linear(200,50),
            nn.ReLU(inplace = True),
            nn.Linear(50, out_size),
        )

    def forward(self, x):
        h = self.feature_extractor(x)
        h = torch.flatten(h,1)
        y = self.classifier(h)
        return y