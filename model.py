import math
import logging
import numpy as np

import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F

from utils import Reshape
from advertorch.utils import NormalizeByChannelMeanStd


# ------------------ Classifiers ------------------- #

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, input_size):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.input_size = input_size

        self.conv1 = conv3x3(input_size[0], nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # hardcoded for now
        last_hid = nf * 8 * block.expansion if input_size[1] in [8,16,21,32,42] else 640
        self.linear = nn.Linear(last_hid, num_classes)
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        self.normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x):
        bsz = x.size(0)
        x = self.normalizer(x)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, *self.input_size))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.return_hidden(x)
        out = self.linear(out)
        return out

def ResNet18(nclasses, nf=20, input_size=(3, 32, 32)):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, input_size)

class MLP(nn.Module):
    def __init__(self, args, num_classes=10, nf=400):
        super(MLP, self).__init__()

        self.input_size = np.prod(args.input_size)
        self.hidden = nn.Sequential(nn.Linear(self.input_size, nf),
                                    nn.ReLU(True),
                                    nn.Linear(nf, nf),
                                    nn.ReLU(True))

        self.linear = nn.Linear(nf, num_classes)

    def return_hidden(self,x):
        x = x.view(-1, self.input_size)
        return self.hidden(x)

    def forward(self, x):
        out = self.return_hidden(x)
        return self.linear(out)