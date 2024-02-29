import torch
import torch.nn as nn
import config

class Darknet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = config.B * 5 + config.C
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        layers = [
            self.convolutional_block(3, 64, kernel_size=7, stride=2, padding=3),
            self.max_pool,
            self.convolutional_block(64, 192, kernel_size=3, stride=1, padding=1),
            self.max_pool,
            self.convolutional_block(192, 128, kernel_size=1, stride=1),
            self.convolutional_block(128, 256, kernel_size=3, stride=1, padding=1),
            self.convolutional_block(256, 256, kernel_size=1, stride=1),
            self.convolutional_block(256, 512, kernel_size=3, stride=1, padding=1),
            self.max_pool
        ]
        
        for i in range(4):
            layers += [
                self.convolutional_block(512, 256, kernel_size=1, stride=1),
                self.convolutional_block(256, 512, kernel_size=3, stride=1, padding=1),
            ]
            
        layers += [
            self.convolutional_block(512, 512, kernel_size=1, stride=1),
            self.convolutional_block(512, 1024, kernel_size=3, stride=1, padding=1),
            self.max_pool
        ]
        
        for i in range(2):
            layers += [
                self.convolutional_block(1024, 512, kernel_size=1, stride=1),
                self.convolutional_block(512, 1024, kernel_size=3, stride=1, padding=1),
            ]
        
        layers += [
            self.convolutional_block(1024, 512, kernel_size=3, stride=1, padding=1),
            self.convolutional_block(512, 1024, kernel_size=3, stride=2, padding=1),
            self.convolutional_block(1024, 512, kernel_size=3, stride=1, padding=1),
            self.convolutional_block(512, 1024, kernel_size=3, stride=1, padding=1),
        ]
        
        self.backbone = nn.Sequential(*layers)
    
    def convolutional_block(self, in_channels, out_channels, kernel_size, stride, padding):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(),
            nn.LeakyReLU(negative_slope=0.1)
        )
        
        return conv_block
    
    def forward(self, x):
        return self.backbone.forward(x)

        