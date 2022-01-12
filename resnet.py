import torch
import torch.nn as nn

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
        return x.view(n, k, self.output_size)

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
		self.conv1 = conv(3, 32, 9, l1, l2, weight_net)
		self.conv2 = conv(32, 32, 9, l2, l2, weight_net)
		self.conv3 = conv(32, 64, 9, l2, l2, weight_net)
		self.pool = MaxPoolNG(l2, l3)
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

		x = seelf.pool(x)
		return x
