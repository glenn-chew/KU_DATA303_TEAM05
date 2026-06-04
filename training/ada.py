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
      use_filtering=False,
      use_frequency=False,
      use_cutout=False,
      use_adaptive_cutout=False,
  ):

    self.p = augment_p
    self.ada_target = ada_target
    self.ada_kimg = ada_kimg
    self.ada_interval = ada_interval
    self.use_color = use_color
    self.use_geometric = use_geometric
    self.use_filtering = use_filtering
    self.use_frequency = use_frequency
    self.use_cutout = use_cutout
    self.use_adaptive_cutout = use_adaptive_cutout

    self.ada_stats = None
    self.batch_count = 0

  def _augment_filtering(self, img):
    # gaussian blur
    if random.random() < 0.5:
      kernel_size = random.choice([3, 5])
      sigma = random.uniform(0.1, 2.0)
      img = TF.gaussian_blur(img, kernel_size, sigma)
    # sharpening
    else:
      sharpness = random.uniform(0.5, 2.0)
      img = TF.adjust_sharpness(img, sharpness)
    return img
  
  def _augment_cutout(self, img):
    h, w = img.shape[1], img.shape[2]

    #fixed at 0.5
    cut_size = int(0.5 * min(h,w))
    cut_size = (max(4, cut_size)) #min 4px

    x1 = randomrandint(0, w - cut_size)
    y1 = randomrandint(0, h - cut_size)
    img = img.clone()
    img[:, y1:y1+cut_size, x1:x1+cut_size] = 0
    return img

  def _augment_adaptive_cutout(self, img):
    h, w = img.shape[1], img.shape[2]

    #scale size with ada_p
    min_size = 0.1
    max_size = 0.5
    cut_ratio = min_size + (max_size - min_size) * self.p
    cut_size = int(cut_ratio * min(h,w))
    cut_size = max(4, cut_size) #minimum 4px

    x1 = randomrandint(0, w - cut_size)
    y1 = randomrandint(0, h - cut_size)
    img = img.clone()
    img[:, y1:y1+cut_size, x1:x1+cut_size] = 0
    return img
  
  # def _augment_frequency(self, img):
  #   # convert to frequency domain
  #   fft = torch.fft.fft2(img)
  #   fft_shifted = torch.fft.fftshift(fft)

  #   # create high frequency mask
  #   h, w = img.shape[1], img.shape[2]
  #   center_h, center_w = h // 2, w // 2
  #   radius = min(h, w) // 3  # low frequency radius

  #   y, x = torch.meshgrid(
  #     torch.arange(h, device=img.device),
  #     torch.arange(w, device=img.device),
  #     indexing='ij'
  #   )
  #   dist = ((y - center_h) ** 2 + (x - center_w) ** 2).sqrt()

  #   # attenuate high frequencies
  #   attenuation = random.uniform(0.3, 0.7)
  #   freq_mask = torch.where(dist > radius,
  #                           torch.ones_like(dist) * attenuation,
  #                           torch.ones_like(dist))

  #   fft_shifted = fft_shifted * freq_mask.unsqueeze(0)
  #   fft = torch.fft.ifftshift(fft_shifted)
  #   img = torch.fft.ifft2(fft).real
  #   img = img.clamp(0, 1)
  #   return img

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
      
      if self.use_filtering:
        img = self._augment_filtering(img)
      if self.use_frequency:
        img = self._augment_frequency(img)
      
      if self.use_adaptive_cutout
        img = self._augment_adaptive_cutout(img)
      if self.use_cutout
        img = self._augment_cutout(cutout)
      
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