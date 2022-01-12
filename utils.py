import numpy as np
import torch 


def cart2pol(coords):
    x = coords[:,0]
    y = coords[:,1]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.stack([theta, rho],1)

def pol2cart(coords):
    theta = coords[:,0]
    rho = coords[:,1]
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.stack([x, y], 1)

def fibonacci_sunflower(n_nodes):
    g_ratio = (np.sqrt(5) + 1) / 2
    nodes = np.arange(1,n_nodes+1)
    rho = np.sqrt(nodes-0.5)/np.sqrt(n_nodes)
    theta = np.pi * 2 * g_ratio * nodes
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return normalize(np.array([x,y]).T)

def fibonacci_retina(n_nodes, fovea, foveal_density):
    x = fibonacci_sunflower(n_nodes)
    x = normalize(x)
    x = cart2pol(x)
    x[:,1] *= (1/(fovea + ((2*np.pi*fovea)/foveal_density)) ** x[:,1] ** foveal_density)
    x = pol2cart(x)
    return torch.tensor(normalize(x), dtype=torch.float32)

def normalize(points):
    points = cart2pol(points)
    points[:,1] /= np.max(points[:,1])
    points = pol2cart(points)
    return points

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)