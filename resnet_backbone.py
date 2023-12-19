import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

"""
Making custom resnet backbone for FasterRCNN object detection model
Steps:
    -- remove layer0
    -- remove first few blocks in layer1
    -- add bottleneck in layer1
    -- remove avgpool, fc layers at end (bc we only want the feature map, not a classification)
"""
class ResidualBlock50(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock50, self).__init__()
        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
                        torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(
                        torch.nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
                        torch.nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = torch.nn.ReLU()
        self.out_channels = out_channels

        print(self.conv1[0].weight.shape)
        print(self.conv2[0].weight.shape)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class encoder(nn.Module):
    def __init__(self, bottleneck_channel=12):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.b1 = nn.BatchNorm2d(64)
        # self.r1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        # self.r2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = F.relu(x, inplace=True)
        x = self.mp1(x)
        x = self.b2(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        return x
        

class decoder(nn.Module):
    def __init__(self, bottleneck_channel=12):
        self.b1 = nn.BatchNorm2d(bottleneck_channel)
        # self.r1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(512)
        # self.r2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(512)
        # self.r3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False)
        self.b4 = nn.BatchNorm2d(512)
        # self.r4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False)
        self.ap1 = nn.AvgPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x = self.b1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        x = self.b2(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.b3(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = self.b4(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = self.ap1(x)
        return x




class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        torch.nn.BatchNorm2d(64),
                        torch.nn.ReLU())
        print(self.conv1[0].weight.shape)
        self.maxpool = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        # self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = torch.nn.AvgPool2d(7, stride=1)
        self.fc = torch.nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            if i == 0:
                pass
                layers.append(self.add_bottleneck(planes))       # BOTTLENECK added here
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)
    
    def add_bottleneck(self, planes):
        pass
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        # x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        print("layer 3 output:", x.shape)   # torch.Size([1, 512, 7, 7])

        # x = self.avgpool(x)               # no need for these, we want a backbone (just feature maps), not a classifier (10 probabilities)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x
    
if __name__ == "__main__":
    import torch
    import torchvision
    input_tensor = torch.randn(1, 3, 224, 224)
    model = ResNet(ResidualBlock50, [3, 4, 6, 3])
    output = model(input_tensor)