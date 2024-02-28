import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.append('..')
from scripts.utils import calc_shape


class ConvNet(torch.nn.Module):
    def __init__(self, kernel_size: int = 5, stride: int = 1, 
                 padding: int = 1, out_channels: int = 32,  pixel: int = 28,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size  
        self.stride = stride
        self.padding = padding
        self.conv1 = torch.nn.Conv2d(1, out_channels, kernel_size=kernel_size, 
                                     stride=stride, padding=padding)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                                     stride=stride, padding=padding)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                                     stride=stride, padding=padding)
        self.pixel_size_conv1 = calc_shape(torch.zeros(1, out_channels, pixel, pixel), kernel_size, stride, padding)[2]
        self.pixel_size_conv2 = calc_shape(torch.zeros(1, out_channels, self.pixel_size_conv1, self.pixel_size_conv1), kernel_size, stride, padding)[2]
        self.pixel_size_conv3 = calc_shape(torch.zeros(1, out_channels, self.pixel_size_conv2, self.pixel_size_conv2), kernel_size, stride, padding)[2]
        
        self.pixel_size = self.pixel_size_conv3
        self.fc = torch.nn.Linear(self.pixel_size*self.pixel_size*out_channels, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x