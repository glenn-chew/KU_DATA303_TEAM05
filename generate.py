# generate.py
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

from networks.generator import Generator

# config
Z_DIM = 512
IMAGE_SIZE = 64
CKPT_PATH = "/home/elicer/KU_DATA303_TEAM05/checkpoints/stylegan2_final.pt"
# CKPT_PATH = "/home/elicer/KU_DATA303_TEAM05/checkpoints/stylegan2_1000kimgs.pt"
OUTPUT_DIR = "/home/elicer/KU_DATA303_TEAM05/samples"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# load EMA generator
generator = Generator(z_dim=Z_DIM, img_resolution=IMAGE_SIZE).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
generator.load_state_dict(ckpt["ema"])
generator.eval()

# compute w mean for truncation
generator.compute_w_mean(n_samples=10000, device=device)

# 1. sample grids at different truncation values
z = torch.randn(16, Z_DIM, device=device)  # fixed z for fair comparison
for psi in [0.5, 0.7, 1.0]:
    with torch.no_grad():
        imgs = generator(z, truncation_psi=psi)
        imgs = (imgs.clamp(-1, 1) + 1) / 2
    vutils.save_image(imgs, f"{OUTPUT_DIR}/grid_psi{psi}.png", nrow=4, padding=2)
    print(f"✓ Saved grid_psi{psi}.png")

# 2. latent interpolation (shows smooth latent space)
z1 = torch.randn(1, Z_DIM, device=device)
z2 = torch.randn(1, Z_DIM, device=device)
alphas = torch.linspace(0, 1, 8, device=device)
interp_imgs = []
with torch.no_grad():
    for a in alphas:
        z_interp = (1 - a) * z1 + a * z2
        img = generator(z_interp, truncation_psi=0.7)
        img = (img.clamp(-1, 1) + 1) / 2
        interp_imgs.append(img)
interp_imgs = torch.cat(interp_imgs, dim=0)
vutils.save_image(interp_imgs, f"{OUTPUT_DIR}/interpolation.png", nrow=8, padding=2)
print("✓ Saved interpolation.png")
