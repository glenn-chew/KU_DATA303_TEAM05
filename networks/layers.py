import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class EqualLinear(nn.Module):
  def __init__(self, in_features, out_features, bias=True, bias_init=0, lr=1):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(out_features, in_features).div(lr))

    if bias:
      self.bias = nn.Parameter(torch.zeros(out_features).fill_(bias_init))
    else:
      self.bias = None

    self.scale = (1 / np.sqrt(in_features)) * lr

  def forward(self, x):
    return F.linear(x, self.weight * self.scale, self.bias)
  
class PixelNorm(nn.Module):
  def __init__(self):
    super(PixelNorm, self).__init__()
    self.epsilon = 1e-8
  def forward(self, x):
    return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class InjectNoise(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(1, channels, 1, 1))

  def forward(self, x):
    noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
    return x + self.weight * noise

class ModulatedConv2d(nn.Module):
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size=3,
      style_dim=512,
      upsample=False,
      downsample=False,
      demodulate=True,
      blur_kernel=[1,3,3,1]
      ):
    super().__init__()

    self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
    self.eps = 1e-8
    self.upsample = upsample
    self.downsample = downsample
    self.demodulate = demodulate
    self.stylemod = EqualLinear(style_dim, in_channels, bias_init=1)

    if upsample or downsample:
      factor = 2
      self.register_buffer('blur_kernel', self._make_blur_kernel(blur_kernel))

    self.padding = kernel_size // 2

  def _make_blur_kernel(self, blur_kernel):
    kernel = torch.tensor(blur_kernel, dtype=torch.float32)
    kernel = kernel[:, None] * kernel[None, :]
    kernel = kernel / kernel.sum()

    return kernel[None, None]

  def _blur(self, x):
    if not hasattr(self, 'blur_kernel'):
      return x

    B, C, H, W = x.shape

    kernel = self.blur_kernel.repeat(C, 1, 1, 1)

    pad = (self.blur_kernel.shape[-1] - 1) // 2
    x = F.conv2d(x, kernel, padding=pad, groups=C)

    return x

  def forward(self, x, style):
    B, Cin, H, W = x.shape

    #Obtain style after linear layer
    style = self.stylemod(style)
    style = style.view(B, 1, Cin, 1, 1) #[B, 1, Cin, 1, 1]

    #Modulation
    weight = self.weight.unsqueeze(0) #[1, Cout, Cin, K, K]
    weight = weight * style #[B, Cout, Cin, K, K]

    #Demodulation
    if self.demodulate:
      demod = torch.rsqrt((weight ** 2).sum([2,3,4]) + self.eps)
      demod = demod.view(B, -1, 1, 1, 1) #[B, Cout, 1, 1, 1]
      weight = weight * demod

    x = x.view(1, B * Cin, H, W)

    #[B*Cout, Cin, K, K]
    weight = weight.view(
        B * weight.shape[1],
        weight.shape[2],
        weight.shape[3],
        weight.shape[4]
        )

    #Conv3x3, K = 3
    if self.upsample:
      out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=B)
      out = self._blur(out)
    elif self.downsample:
      out = self._blur(x)
      out = F.conv2d(out, weight, padding=0, stride=2, groups=B)
    else:
      out = F.conv2d(x, weight, padding=self.padding, groups=B) #[1, B*Cout, H, W]

    _, _, H_out, W_out = out.shape
    return out.view(B, -1, H_out, W_out)
  
class ToRGB(nn.Module):
  def __init__(self, in_channels, style_dim, upsample=False):
    super().__init__()
    self.upsample = upsample
    self.conv = ModulatedConv2d(
        in_channels,
        3,
        kernel_size=1,
        style_dim=style_dim,
        demodulate=False
        )
    self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

  def forward(self, x, style, skip=None):
    x = self.conv(x, style)
    x = x + self.bias
    if skip is not None:
      if self.upsample:
        skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
      x = x + skip

    return x

class StyleBlock(nn.Module):
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      style_dim,
      upsample=False,
      blur_kernel=[1,3,3,1]
      ):
    super().__init__()

    self.conv = ModulatedConv2d(
        in_channels,
        out_channels,
        kernel_size,
        style_dim,
        upsample=upsample,
        blur_kernel=blur_kernel)

    self.noise = InjectNoise(out_channels)
    self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
    self.activation = nn.LeakyReLU(0.2)

  def forward(self, x, style):
    x = self.conv(x, style)
    x = self.noise(x)
    x = x + self.bias
    x = self.activation(x)

    return x
