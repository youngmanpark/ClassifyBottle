import torch.nn as nn
import torch.nn.functional as F
import torch
from ResidualBlock import ResidualBlock 

class Model(nn.Module):
    def __init__(self,num_blocks,num_classes=4):
        super(Model,self).__init__()
        self.in_channels = 64
        self.base = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(64,num_blocks[0],stride=2)
        self.layer2 = self._make_layer(128,num_blocks[1],stride=2)
        self.layer3 = self._make_layer(256,num_blocks[2],stride=2)
        self.layer4 = self._make_layer(512,num_blocks[3],stride=2)
        self.gap = nn.AvgPool2d(4)
        self.fc1 = nn.Linear(8192,100)
        self.fc2 = nn.Linear(100,4)

    
    def _make_layer(self,out_channels,num_blocks,stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            block = ResidualBlock(self.in_channels,out_channels,stride)
            layers.append(block)
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out