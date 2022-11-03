import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride=1):
        super(BasicBlock,  self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential() # identity인 경우 
        if stride != 1 :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # skip connection 
        out = F.relu(out)
        return out 
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.layer1_1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2_1 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_1 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4_1 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(1024, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        fin = np.fft.fft2(x)
        fshift = np.fft.fftshift(fin)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        fout = F.relu(self.bn1_1(self.conv1_1(fin)))
        fout = self.layer1_1(fout)
        fout = self.layer2_1(fout)
        fout = self.layer3_1(fout)
        fout = self.layer4_1(fout)
        fout = F.avg_pool2d(fout, 4)
        fout = fout.view(fout.size(0), -1)

        out = self.linear(out + fout)
        return out 

def FResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])