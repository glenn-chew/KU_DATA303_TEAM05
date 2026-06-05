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
      use_blit=True,
  ):

    self.p = augment_p
    self.ada_target = ada_target
    self.ada_kimg = ada_kimg
    self.ada_interval = ada_interval
    self.use_color = use_color
    self.use_geometric = use_geometric
    self.use_blit = use_blit

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

  def _augment_geom(self, img):
    h, w = img.shape[1], img.shape[2]

    # isotropic scaling
    if random.random() < self.p:
      s = 2 ** (random.gauss(0, 0.2))
      new_h = max(4, int(h * s))
      new_w = max(4, int(w * s))
      img = TF.resize(img, [new_h, new_w], antialias=True)
      img = TF.center_crop(img, [h, w]) if s > 1 else TF.pad(
          img, [(w - new_w)//2, (h - new_h)//2,
                (w - new_w+1)//2, (h - new_h+1)//2]
      )

    # arbitrary rotation
    if random.random() < self.p:
      angle = random.uniform(-180, 180)
      img = TF.rotate(img, angle)

    # fractionla translation
    if random.random() < self.p:
      tx_frac = random.gauss(0, 0.125) * w
      ty_frac = random.gauss(0, 0.125) * h
      img = TF.affine(img, angle=0, translate=[tx_frac, ty_frac],
                      scale=1.0, shear=0)

    return img
  
  def _augment_color(self, img):
    # brightness
    if random.random() < self.p:
      b = random.gauss(0, 0.2)
      img = (img + b).clamp(0, 1)

    # contrast
    if random.random() < self.p:
      c = 2**random.gauss(0, 0.5)
      img = ((img-0.5) * c + 0.5).clamp(0, 1)

    # luma flip
    if random.random() < self.p:
      i = random.randint(0, 1)
      if i == 1:
        img = 1.0 - img
      
    # hue rotation
    if random.random() < self.p:
      angle = random.uniform(-180, 180)
      img = TF.adjust_hue(img, angle/360)
    
    # saturation
    if random.random() < self.p:
      s = 2 ** random.gauss(0, 1.0)
      img = TF.adjust_saturation(img, s)

    return img.clamp(0, 1)


    

 
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
        img = self._augment_color(img)
      if self.use_geometric:
        img = self._augment_geom(img)
      if self.use_blit:
        img = self._augment_blit(img)
      
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