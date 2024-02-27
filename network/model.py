import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.append('..')

from scripts.utils import  calc_shape

class ConvNet(torch.nn.Module):
    def __init__(self, kernel_size: int = 3, stride: int = 1, padding: int = 1, out_channels: int = 32, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size  
        self.stride = stride
        self.padding = padding
        self.conv1 = torch.nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = torch.nn.Linear(1 * 26 * 26, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x