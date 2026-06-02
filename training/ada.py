import torch
import torchvision.transforms.functional as TF
import random

class ADA:
  def __init__(
      self,
      augment_p=0.0,
      ada_target=0.6,
      ada_kimg=500,
      ada_interval=4,
      use_color=True,
      use_geometric=True,
      use_filtering=True,
  ):

    self.p = augment_p
    self.ada_target = ada_target
    self.ada_kimg = ada_kimg
    self.ada_interval = ada_interval
    self.use_color = use_color
    self.use_geometric = use_geometric
    self.use_filtering = use_filtering

    self.ada_stats = None
    self.batch_count = 0


  def __call__(self, images):
    # images: [B,3,H,W], augmented: [B,3,H,W]
    if self.p <= 0:
      return images

    B = images.shape[0]
    augment_mask = torch.rand(B, device=images.device) < self.p

    if not augment_mask.any():
      return images

    # rescale to [0, 1] for transforms
    in_neg_one = images.min() < 0
    images_01 = (images + 1.0) / 2.0 if in_neg_one else images

    augmented = images_01.clone()
    selected = images_01[augment_mask]

    # process each image individually
    result = []
    for img in selected:
      if self.use_color:
        img = TF.adjust_brightness(img, 1 + random.uniform(-0.2, 0.2))
        img = TF.adjust_contrast(img, 1 + random.uniform(-0.2, 0.2))
        img = TF.adjust_saturation(img, 1 + random.uniform(-0.2, 0.2))
        img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))
      if self.use_geometric:
        if random.random() < 0.5:
          img = TF.hflip(img)
        angle = random.uniform(-15, 15)
        img = TF.rotate(img, angle)
      result.append(img)

    augmented[augment_mask] = torch.stack(result)

    if in_neg_one:
      augmented = augmented * 2.0 - 1.0

    return augmented

  def update_p(self, rt_sign):

    # if Discriminator is to confident on real images,
    # increase augmentation probability
    adjust = rt_sign - self.ada_target
    adjust = adjust * (self.ada_interval / (self.ada_kimg * 1000))

    # update and clamp
    self.p = max(0.0, min(1.0, self.p + adjust))
    self.batch_count += 1