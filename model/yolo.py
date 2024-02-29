import torch
import torch.nn as nn
import config
from backbone import *

class Yolov1(nn.Module):
    def __init__(self, backbone=Darknet):
        self.backbone = backbone
        self.depth = config.B * 5 + self.C    
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.S * config.S * 1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, config.S * config.S * self.depth)
        )
        
        
    def forward(self, x):
        x = self.backbone(x)
        return torch.reshape(
            self.head(x),
            (x.size(dim=0), config.S, config.S, self.depth)
        )
    