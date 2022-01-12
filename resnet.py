import torch
import torch.nn as nn
from layers import *
from utils import *

class WeightNet(nn.Module):
    def __init__(self, output_size, hidden_size, omega):
        super().__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_size)
        self.omega = omega
    
    def forward(self, x):
        n, k, _ = x.shape
        x = torch.sin(self.fc1(x) * self.omega)
        x = self.fc2(x)
        x = x.view(n, k, self.output_size)
        return x

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, in_locs, out_locs, weight_net):
        super().__init__()

        conv = InterpConv
        self.conv1 = conv(inplanes, planes, 9, in_locs, out_locs, weight_net)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv(planes, planes, 9, out_locs, out_locs, weight_net)
        self.bn2 = nn.BatchNorm1d(planes)
        self.activation = nn.ReLU(inplace=True)

        self.downsample = None
        self.expand = None
        if len(in_locs) != len(out_locs):
            self.downsample = AvgPoolNG(4, in_locs, out_locs)
        if inplanes != planes:
            self.expand = nn.Conv1d(inplanes, planes, 1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.expand is not None:
            identity = self.expand(identity)
        out += identity
        out = self.activation(out)
        return out

class Stem(nn.Module):
    def __init__(self, l1, l2, l3, weight_net):
        super().__init__()
        conv = InterpConv
        self.conv1 = conv(3, 32, 9, l1, l2, weight_net)
        self.conv2 = conv(32, 32, 9, l2, l2, weight_net)
        self.conv3 = conv(32, 64, 9, l2, l2, weight_net)
        self.pool = MaxPoolNG(4, l2, l3)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.pool(x)
        return x

class ResNet18NG(nn.Module):
    def __init__(self, input_locs, n_classes):
        super().__init__()
        l1 = torch.tensor(input_locs, dtype=torch.float32).clone()
        l2 = torch.tensor(fibonacci_retina(2048, 0.1, 1.6), dtype=torch.float32).clone().detach() #* 0.9
        l3 = torch.tensor(fibonacci_retina(512, 0.1, 1.6) , dtype=torch.float32).clone().detach()# * 0.8
        l4 = torch.tensor(fibonacci_retina(128, 0.1, 1.6), dtype=torch.float32).clone().detach() #* 0.7
        l5 = torch.tensor(fibonacci_retina(32, 0.1, 1.6), dtype=torch.float32).clone().detach() #* 0.6
        l6 = torch.tensor(fibonacci_retina(9, 0.1, 1.6), dtype=torch.float32).clone().detach() #*0.5

        self.filter_network = WeightNet(9, 64, 6)
        self.stem = Stem(l1, l2, l3, self.filter_network)
        self.block1_1 = BasicBlock(64, 64, l3, l3, self.filter_network)
        self.block1_2 = BasicBlock(64, 64, l3, l3, self.filter_network)

        self.block2_1 = BasicBlock(64, 128, l3, l4, self.filter_network)
        self.block2_2 = BasicBlock(128, 128, l4, l4, self.filter_network)

        self.block3_1 = BasicBlock(128, 256, l4, l5, self.filter_network)
        self.block3_2 = BasicBlock(256, 256, l5, l5, self.filter_network)
        
        self.block4_1 = BasicBlock(256, 512, l5, l6, self.filter_network)
        self.block4_2 = BasicBlock(512, 512, l6, l6, self.filter_network)
        
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(512, n_classes)
    
    def forward(self, x):
        x = self.stem(x)

        x = self.block1_1(x)
        x = self.block1_2(x)

        x = self.block2_1(x)
        x = self.block2_2(x)

        x = self.block3_1(x)
        x = self.block3_2(x)

        x = self.block4_1(x)
        x = self.block4_2(x)

        x = self.gap(x)
        x = x.view(-1, 512)
        x = self.fc(x)

        return x