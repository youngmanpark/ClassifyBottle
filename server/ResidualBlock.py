import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResidualBlock,self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,stride = stride,
            padding=1,bias=False),nn.BatchNorm2d(self.out_channels),nn.ReLU(),
            nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,stride = 1,
            padding=1,bias=False),nn.BatchNorm2d(self.out_channels))
        if self.stride != 1 or self.in_channels != self.out_channels:
            self.downsample = nn.Sequential(
            nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,stride = stride,
            padding=1,bias=False),nn.BatchNorm2d(self.out_channels)
            )
    
    def forward(self,x):
        out = self.conv_block(x)
        

        if self.stride != 1 or self.in_channels != self.out_channels:
            x = self.downsample(x)
        out = F.relu(x + out)

        return out