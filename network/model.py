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

        self.fc1 = torch.nn.Linear(1 * 26 * 26, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = x.flatten(start_dim=1)
        x = x.view(x.size(0), -1)
        # [batch_size, out_channels, 28, 28]
        # print(x.shape)
        x = self.fc1(x)  # Uncomment if needed
        # print(F.softmax(x, dim=1))  # Use for multi-class classification
        # x = x.argmax(dim=1)  # Uncomment if needed
        # breakpoint()    
        # print("goes here")
        return x#.argmax(dim=1)  # Reshape if needed