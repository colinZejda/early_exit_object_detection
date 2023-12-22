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

Training
1) make teacher resnet (train on imagenet dataset), then student is phsyically split resnet
    -- phys split: train head + encoder/decoder, freeze tail
    -- phys split: reverse the freeze, only train tail
    -- now we have fully trained reg resnet50, and phys split resnet50
2) now, train 2 models
    -- time for obj det model, don't train backbone (freeze it), only obj det heads
    -- model 1: reg obj det
    -- model 2: obj det model + phys split backbone
3) add early exit
    -- determine whether we should go thru with the split (is our pred (bb + class) good enough yet?)
    -- perform another round of knowledge distillation, train up the early exit
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

###################################################################

class encoder(nn.Module):
    def __init__(self, bottleneck_channel=12):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.b1 = nn.BatchNorm2d(64)
        # self.r1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        # self.r2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = F.relu(x, inplace=True)
        x = self.mp1(x)
        x = self.b2(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        return x
        
###################################################################

class decoder(nn.Module):
    def __init__(self, bottleneck_channel=12):
        super(decoder, self).__init__()
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
        self.conv4 = nn.Conv2d(512, 64, kernel_size=2, stride=1, bias=False)
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

###################################################################

class ResNetHead(torch.nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNetHead, self).__init__()
        self.inplanes = 64
        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        torch.nn.BatchNorm2d(64),
                        torch.nn.ReLU())
        self.encoder = encoder()
        self.mp = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
    def forward(self, x):
        x = self.conv1(x)
        print("head after conv1", x.shape)   # torch.Size([1, 64, 112, 112])  --> batched, channels, height, we
        x = self.encoder(x)
        print("head after encoder", x.shape) # torch.Size([1, 12, 15, 15])
        x = self.mp(x)                       # maxpool after encoder, gets better compression 
        print("head after maxpool", x.shape) # torch.Size([1, 12, 8, 8]) 
        return x

###################################################################

class ResNetTail(torch.nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNetTail, self).__init__()
        self.inplanes = 64
        self.decoder = decoder()
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2, cut_blocks=True)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)

    def forward(self, x):
        print("tail start x", x.shape)       # torch.Size([1, 12, 8, 8])
        x = self.decoder(x)
        print("tail after decoder", x.shape) # torch.Size([1, 512, 7, 7])
        # x = x.view(1, 64, 1568)
        # print("x after reshaping for L2", x.shape)    # 
        x = self.layer2(x)
        print("tail after L2", x.shape)      #
        x = self.layer3(x)
        print("tail after L3", x.shape)      #
        return x
    
    def _make_layer(self, block, planes, blocks, stride=1, cut_blocks=False):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        start = 3 if cut_blocks else 1
        for i in range(start, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)
    

if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 224, 224)
    head = ResNetHead(ResidualBlock50, [3, 4, 6, 3])
    tail = ResNetTail(ResidualBlock50, [3, 4, 6, 3])
    h_out = head(input_tensor)
    t_out = tail(h_out)
    print("Backbone OUT", t_out.shape)