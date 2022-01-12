import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree


def support(points, sample_locs, k):
    tree = cKDTree(points)
    dist, idx = tree.query(sample_locs, k)
    points = points[idx,:]

    # calculate each pixel's spatial offsets from center of its patch
    points = points - sample_locs.unsqueeze(1)
    # Normalize the patch to lie within a unit circle
    new_points = points / dist.max(axis=1)[:,np.newaxis, np.newaxis]
    return new_points, torch.tensor(idx, dtype=torch.long)

class MaxPoolNG(nn.Module):
    def __init__(self, k, locs_in, locs_out):
        super().__init__()
        self.register_buffer('idx', support(locs_in, locs_out, k)[1])
    
    def forward(self, x):
        x = x[:,:, self.idx]
        x = x.max(-1)[0]
        return x

class AvgPoolNG(nn.Module):
    def __init__(self, k, locs_in, locs_out):
        super().__init__()
        self.register_buffer('idx', support(locs_in, locs_out, k)[1])
    
    def forward(self, x):
        x = x[:,:, self.idx]
        x = x.mean(-1)
        return x

class InterpConv(nn.Module):
    def __init__(self, in_channels, out_channels, k, locs_in, locs_out, weight_net):
        super().__init__()

        # support regions / image patches and each pixel's corresponding spatial offset
        # from the center of the patch
        self.s_locs, self.s_idx = support(locs_in, locs_out, self.k)
        self.s_locs = nn.Parameter(self.s_locs.type(torch.float32), requires_grad=True)
        self.register_buffer('unfold_idx', self.s_idx)

        # Network that computes filter weights from spatial coordinates
        self.weight_net = weight_net
        
        # Conv Kernel
        self.kernel = nn.Parameter(torch.rand((out_channels, in_channels, k)))
        nn.init.xavier_uniform_(self.kernel)

        # TODO: Remove this batch norm by sorting by normalizing output of weight_net
        # Batch Norm to manage expliding gradients
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # 'Unfold' image
        x = x[ :, :, self.unfold_idx]
        B, I, N, K = x.shape

        # Compute interpolation kernels using the weight_net
        weights = self.weight_net(self.s_locs.clone().detach().requires_grad_(True))
        weights = weights.view(N, K, K)

        # b: batch
        # i: in_channels
        # o: out_channels
        # n: size of feature map / number of pixels
        # k: interpolation kernel size
        # s: conv kernel size

        # Interpolate
        x = torch.einsum('bink,nks->bins', x, weights)
        # Convolve
        x = torch.einsum('bins,ois->bon', x, self.kernel)

        x = self.bn(x)

        return x