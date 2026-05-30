import kornia.augmentation as K
import torch

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

    self.color_aug = K.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.1,
        p=1.0)
    self.geometric_aug = K.AugmentationSequential(
      K.RandomHorizontalFlip(p=0.5),
      K.RandomAffine(
          degrees=0,
          translate=(0.125,0.125),
          scale=(0.8,1.2),
          p=1.0
      ),
      K.RandomRotation(degrees=10, p=0.5),
    )
    self.filtering_aug = K.AugmentationSequential(
        K.RandomGaussianBlur(kernel_size=(3,3), sigma=(0.1,2.0), p=0.5),
        K.RandomSharpness(sharpness=0.5, p=0.5),
    )

  def _augment_color(self, images):
    return self.color_aug(images)
  def _augment_geometric(self, images):
    return self.geometric_aug(images)
  def _augment_filtering(self, images):
    return self.filtering_aug(images)


  def __call__(self, images):
    # images: [B,3,H,W], augmented: [B,3,H,W]
    if self.p <= 0:
      return images

    batch_size = images.shape[0]

    # randomly select which images to augmenta nd create mask for each image in batch
    augment_mask = torch.rand(batch_size, device=images.device) < self.p

    # rescale to [0, 1] for Kornia
    if images.min() < 0:
      images1 = (images + 1.0) / 2.0
    else:
      images1 = images

    # clone images to prevent modifying in-place
    augmented = images1.clone()

    # apply mask to selected images
    if augment_mask.any():
      selected_images = images1[augment_mask]

      if self.use_color:
        selected_images = self._augment_color(selected_images)

      if self.use_geometric:
        selected_images = self._augment_geometric(selected_images)

      if self.use_filtering:
        selected_images = self._augment_filtering(selected_images)

      augmented[augment_mask] = selected_images

    # convert back to [-1, 1]
    if images.min() < 0:
      augmented = (augmented - 0.5) * 2.0

    return augmented

  def update_p(self, rt_sign):

    # if Discriminator is to confident on real images,
    # increase augmentation probability
    adjust = rt_sign - self.ada_target
    adjust = adjust * (self.ada_interval / (self.ada_kimg * 1000))

    # update and clamp
    self.p = max(0.0, min(1.0, self.p + adjust))
    self.batch_count += 1