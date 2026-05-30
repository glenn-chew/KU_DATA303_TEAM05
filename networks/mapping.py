import torch
import torch.nn as nn
from networks.layers import EqualLinear, PixelNorm

class MappingNetwork(nn.Module):
  def __init__(self, z_dim=512, w_dim=512, lr=0.01):
    super().__init__()
    self.mapping = nn.Sequential(
        PixelNorm(),
        EqualLinear(z_dim, w_dim, lr=lr),
        nn.LeakyReLU(0.2),
        EqualLinear(w_dim, w_dim, lr=lr),
        nn.LeakyReLU(0.2),
        EqualLinear(w_dim, w_dim, lr=lr),
        nn.LeakyReLU(0.2),
        EqualLinear(w_dim, w_dim, lr=lr),
        nn.LeakyReLU(0.2),
        EqualLinear(w_dim, w_dim, lr=lr),
        nn.LeakyReLU(0.2),
        EqualLinear(w_dim, w_dim, lr=lr),
        nn.LeakyReLU(0.2),
        EqualLinear(w_dim, w_dim, lr=lr),
        nn.LeakyReLU(0.2),
        EqualLinear(w_dim, w_dim, lr=lr),
    )

  def forward(self, x):
    return self.mapping(x)