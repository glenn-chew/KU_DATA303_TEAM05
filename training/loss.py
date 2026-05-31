import torch
import torch.nn.functional as F
import numpy as np

class StyleGAN2Loss:
  def __init__(
      self,
      r1_gamma=10.0,
  ):
    self.r1_gamma = r1_gamma

    self.p1_mean = None
  def discriminator_loss(self, real_logits, fake_logits):
    # L_D = -E[log(sigmoid(D(real)))] - E[log(1 - sigmoid(D(fake)))]
       #  = E[softplus(-D(real))] + E[softplus(D(fake))]

    real_loss = F.softplus(-real_logits).mean()
    fake_loss = F.softplus(fake_logits).mean()
    return real_loss + fake_loss

  def generator_loss(self, fake_logits, fake_images=None):
    loss = F.softplus(-fake_logits).mean()
    
    # add channel consistency penalty
    if fake_images is not None:
        r, g, b = fake_images[:,0], fake_images[:,1], fake_images[:,2]
        channel_loss = (r - g).pow(2).mean() + (r - b).pow(2).mean() + (g - b).pow(2).mean()
        loss = loss + 0.1 * channel_loss
    
    return loss

  def r1_penalty(self, real_images, real_logits):
    # compute gradients of D(real) wrt real images
    gradients = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_images,
        create_graph=True,
        # retain_graph=True,
    )[0]

    # compute ||gradient of D||^2
    gradient_penalty = gradients.pow(2).reshape(gradients.shape[0],-1).sum(dim=-1).mean()
    return self.r1_gamma / 2 * gradient_penalty
  
class LossTracker:
  def __init__(self):
    self.losses = {
        'g_loss' : [],
        'd_loss' : [],
        'r1_penalty': [],
        'real_score' : [],
        'fake_score' : []
    }

  def update(self, **kwargs):
    for k, v in kwargs.items():
      if k in self.losses:
        if isinstance(v, torch.Tensor):
          v = v.item()
        self.losses[k].append(v)
  def get_averages(self, window=100):
    averages = {}
    for k, v in self.losses.items():
      if len(v) > 0:
        recent = v[-window:]
        averages[k] = np.mean(recent)
    return averages

  def reset(self):
    for k in self.losses.keys():
      self.losses[k] = []