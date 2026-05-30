import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EqualConv2d(nn.Module):
  def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0):
    super().__init__()

    self.weight = nn.Parameter(
        torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    )
    self.bias = nn.Parameter(torch.zeros(out_channels))
    self.stride = stride
    self.padding = padding

    self.scale = 1 / np.sqrt(in_channels * kernel_size ** 2)

  def forward(self, x):
    return F.conv2d(
        x,
        self.weight * self.scale,
        bias=self.bias,
        stride=self.stride,
        padding=self.padding
    )
  
class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, downsample=True):
    super().__init__()

    self.conv1 = EqualConv2d(in_channels, in_channels, 3, padding=1)
    self.conv2 = EqualConv2d(in_channels, out_channels, 3, padding=1)
    self.skip = EqualConv2d(in_channels, out_channels, 1)
    self.act = nn.LeakyReLU(0.2)
    self.downsample = downsample
  def forward(self, x):
    out = self.conv1(x)
    out = self.act(out)
    out = self.conv2(out)
    out = self.act(out)

    skip = self.skip(x)

    out = (out + skip) / np.sqrt(2)

    if self.downsample:
      out = F.avg_pool2d(out, 2)

    return out 

class FromRGB(nn.Module):
  def __init__(self, out_channels):
    super().__init__()
    self.conv = EqualConv2d(3, out_channels, 1)
    self.act = nn.LeakyReLU(0.2)

  def forward(self, x):
    return self.act(self.conv(x))
  

# 256x256 -> 128x128 ... 4x4 -> predict

class Discriminator(nn.Module):
  def __init__(self,
               img_channels=3,
               img_resolution=256,
               channel_base=32768,
               channel_max=512):
    super().__init__()

    self.img_channels = img_channels
    self.img_resolution = img_resolution

    log_resolution = int(np.log2(img_resolution))
    # self.log_resolution = int(torch.log2(torch.tensor(img_resolution)).item())

    # RGB layer
    init_channels = min(channel_base // img_resolution, channel_max)
    self.from_rgb = FromRGB(init_channels)

    # Progressive residual block (build from img_res down to 4x4)
    self.blocks = nn.ModuleList()
    in_channels = init_channels

    for res in range(log_resolution, 2, -1):
      out_channels = min(channel_base // (2 ** res), channel_max)
      self.blocks.append(ResBlock(in_channels, out_channels, downsample=True))
      in_channels = out_channels

    # Final layers (4x4)
    self.final_conv = EqualConv2d(in_channels, in_channels, 3, padding=1)
    self.final_linear = nn.Linear(in_channels * 4 * 4, in_channels)
    self.output_linear = nn.Linear(in_channels, 1)

    self.act = nn.LeakyReLU(0.2)

  def forward(self, img):
    #img: [B, 3, Res, Res], score: [B, 1] (real/fake)

    # convert to RGB
    x = self.from_rgb(img)

    # Progressive downsample
    for block in self.blocks:
      x = block(x)

    # final layers
    B = x.shape[0]
    x = self.final_conv(x)
    x = self.act(x)

    # flatten and classify
    x = x.view(B, -1)
    x = self.final_linear(x)
    x = self.act(x)

    score = self.output_linear(x)

    return score