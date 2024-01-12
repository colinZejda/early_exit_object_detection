import torch
import torch.nn as nn
import torchvision.models as models

class TeacherResNet(nn.Module):
    def __init__(self):
        super(TeacherResNet, self).__init__()
        resnet50 = models.resnet50(pretrained=True)

        # Example of extracting individual layers
        self.conv1   = resnet50.conv1
        self.bn1     = resnet50.bn1
        self.relu    = resnet50.relu
        self.maxpool = resnet50.maxpool

        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        self.l1_out = None
        self.l2_out = None
        self.l3_out = None
        self.l4_out = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)  # Adjust output features as needed

    def forward(self, x):
        # Define the forward pass, layer by layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        self.l1_out = self.layer1(x)
        self.l2_out = self.layer2(self.l1_out)
        self.l3_out = self.layer3(self.l2_out)
        x           = self.layer4(self.l3_out)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

