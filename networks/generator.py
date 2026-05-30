import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.mapping import MappingNetwork
from networks.layers import StyleBlock, ToRGB, InjectNoise, ModulatedConv2d


# 4x4 conv
# 8x8 upsample conv
# 8x8 conv
# ToRGB
# 16x16 upsample conv
# 16x16 conv
# ToRGB
# ...

class SynthesisNetwork(nn.Module):
  def __init__(self,
               w_dim=512,
               img_channels=3,
               img_resolution=256,
               channel_base=32768,
               channel_max=512):
    super().__init__()
    self.w_dim = w_dim
    self.img_channels = img_channels
    self.img_resolution = img_resolution

    # Calculate number of layres using img_resolution
    self.log_resolution = int(torch.log2(torch.tensor(img_resolution)).item())
    self.num_convs = (self.log_resolution - 2) * 2 + 1
    self.num_torgbs = self.log_resolution - 2
    self.num_layers = self.num_convs + self.num_torgbs


    # constant input
    self.constant = nn.Parameter(torch.randn(1, 512, 4, 4))

    self.convs = nn.ModuleList()
    self.to_rgbs = nn.ModuleList()

    in_channels = 512

    #build synthesis architecture
    # build 4x4 conv separately
    self.convs.append(StyleBlock(in_channels, in_channels, 3, w_dim))

    for res in range(3, self.log_resolution + 1):
      out_channels = min(channel_base // (2 ** res), channel_max)

      self.convs.append(StyleBlock(in_channels, out_channels, 3, w_dim, upsample=True))
      self.convs.append(StyleBlock(out_channels, out_channels, 3, w_dim))
      self.to_rgbs.append(ToRGB(out_channels, w_dim, upsample=(res > 3)))
      in_channels = out_channels

  def forward(self, w, noise='random'):
    B = w.shape[0]

    # w broadcast
    if w.dim() == 2:
      w = w.unsqueeze(1).repeat(1, self.num_layers, 1) # [batch, num_layers, w_dim]

    x = self.constant.repeat(B, 1, 1, 1)

    # 4x4 conv block
    x = self.convs[0](x, w[:, 0])

    #Progressive generation through resolutions
    rgb = None
    conv_idx = 1
    to_rgb_idx = 0
    w_idx = self.num_convs # to_rgb styles start after all conv styles

    for res in range(3, self.log_resolution + 1):
      x = self.convs[conv_idx](x, w[:, conv_idx])
      conv_idx += 1
      x = self.convs[conv_idx](x, w[:, conv_idx])
      conv_idx += 1

      rgb = self.to_rgbs[to_rgb_idx](x, w[:, w_idx], skip=rgb)
      to_rgb_idx += 1
      w_idx += 1

    return torch.tanh(rgb)

class Generator(nn.Module):
  def __init__(
      self,
      z_dim=512,
      w_dim=512,
      img_resolution=256,
      img_channels=3,
      mapping_kwargs={},
      synthesis_kwargs={}
  ):
    super().__init__()

    self.z_dim = z_dim
    self.w_dim = w_dim
    self.img_resolution = img_resolution

    self.mapping = MappingNetwork(z_dim, w_dim, **mapping_kwargs)
    # precompute w_mean from many samples
    self.register_buffer('w_mean', torch.zeros(1, w_dim))

    self.synthesis = SynthesisNetwork(
        w_dim=w_dim,
        img_channels=img_channels,
        img_resolution=img_resolution,
        **synthesis_kwargs)

  def compute_w_mean(self, n_samples=10000, device='cuda'):
    with torch.no_grad():
      z = torch.randn(n_samples, self.z_dim, device=device)
      w_samples = self.mapping(z)
      self.w_mean.copy_(w_samples.mean(0, keepdim=True))

  def forward(self, z, truncation_psi=1.0, noise_mode='random'):
    w = self.mapping(z)

    # Apply truncation trick
    if truncation_psi < 1.0:
      w = self.w_mean + truncation_psi * (w - self.w_mean)

    img = self.synthesis(w, noise=noise_mode)

    return img

  def get_latent(self, z):
    return self.mapping(z)
    
