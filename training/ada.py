import torch
import torchvision.transforms.functional as TF
import random
import numpy as np

class ADA:
  def __init__(
      self,
      augment_p=0.0,
      ada_target=0.6,
      ada_kimg=500,
      ada_interval=4,
      use_color=True,
      use_geometric=True,
      use_blit=True,
      use_ada=True,
  ):

    self.p = augment_p
    self.ada_target = ada_target
    self.ada_kimg = ada_kimg
    self.ada_interval = ada_interval
    self.use_color = use_color
    self.use_geometric = use_geometric
    self.use_blit = use_blit
    self.use_ada = use_ada

    self.ada_stats = None
    self.batch_count = 0

  def _augment_blit(self, img):
    h, w = img.shape[1], img.shape[2]

    # x-flip
    if random.random() < self.p:
      i = random.randint(0, 1)
      if i == 1:
        img = TF.hflip(img)
    
    # 90 degree rotation
    if random.random() < self.p:
      i = random.randint(0,3)
      if i > 0:
        img = torch.rot90(img, i , dims=[1,2])

    # integer translation
    if random.random() < self.p:
      tx = random.uniform(-0.125, 0.125)
      ty = random.uniform(-0.125, 0.125)
      tx_pixels = round(tx*w)
      ty_pixels = round(ty * h)
      img = torch.roll(img, shifts=(ty_pixels, tx_pixels), dims=(1,2))
    
    return img

  def _augment_geometric(self, img):
    # rotation
    angle = random.uniform(-15, 15) if random.random() < self.p else 0 

    # translation
    tx = random.uniform(-0.125, 0.125) * img.shape[2] if random.random() < self.p else 0 
    ty = random.uniform(-0.125, 0.125) * img.shape[1] if random.random() < self.p else 0

    # isotropic scale
    scale = float(np.clip(2 ** random.gauss(0, 0.2), 0.75, 1.33)) if random.random() < self.p else 1.0 
    
    img = TF.affine(
        img,
        angle=angle,
        translate=[tx, ty],
        scale=scale,
        shear=0
    )
    return img
  
  def _augment_color(self, img):
    # brightness
    if random.random() < self.p:
      img = TF.adjust_brightness(img, 1 + random.uniform(-0.2, 0.2))

    # contrast
    if random.random() < self.p:
      img = TF.adjust_contrast(img, 1 + random.uniform(-0.2, 0.2))
      
    # hue rotation
    if random.random() < self.p:
      img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))
    
    # saturation
    if random.random() < self.p:
      img = TF.adjust_saturation(img, 1 + random.uniform(-0.2, 0.2))

    return img.clamp(0, 1)

 
  def __call__(self, images):
    # images: [B,3,H,W], augmented: [B,3,H,W]
    if self.p <= 0:
        return images

    in_neg_one = images.min() < 0
    images_01 = (images + 1.0) / 2.0 if in_neg_one else images

    result = []
    for img in images_01:
        if self.use_blit:
            img = self._augment_blit(img)
        if self.use_geometric:
            img = self._augment_geometric(img)
        if self.use_color:
            img = self._augment_color(img)
        result.append(img)

    augmented = torch.stack(result)
    if in_neg_one:
        augmented = augmented * 2.0 - 1.0
    return augmented

  def update_p(self, rt_sign):
    if not self.use_ada:
      return

    # if Discriminator is to confident on real images,
    # increase augmentation probability
    adjust = rt_sign - self.ada_target
    adjust = adjust * (self.ada_interval / (self.ada_kimg * 1000))

    # update and clamp
    self.p = max(0.0, min(1.0, self.p + adjust))
    self.batch_count += 1