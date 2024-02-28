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
        self.pixel_size_conv1 = calc_shape(torch.zeros(1, out_channels, pixel, pixel), kernel_size, stride, padding)[2]
        self.pixel_size_conv2 = calc_shape(torch.zeros(1, out_channels, self.pixel_size_conv1, self.pixel_size_conv1), kernel_size, stride, padding)[2]
        self.pixel_size = self.pixel_size_conv2
        self.fc = torch.nn.Linear(self.pixel_size*self.pixel_size*out_channels, 10)
        # self.fc1 = torch.nn.Linear(1 * 26  * 26, 10)

    def forward(self, x):
        # [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))
        # [batch_size, 32, 24, 24]
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        # breakpoint()
        # x = torch.nn.Linear(x.shape[-1], 10)(x)
        x = self.fc(x)
        return x