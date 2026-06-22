import os
import torch
import torchvision.utils as vutils
from networks.generator import Generator
from cleanfid import fid

CHECKPOINTS = [
    ("1000kimgs",  "./baseline_eval/stylegan2_1000kimgs.pt"),
    ("5000kimgs",  "./baseline_eval/stylegan2_5000kimgs.pt"),
    ("10000kimgs", "./baseline_eval/stylegan2_10000kimgs.pt"),
    ("15000kimgs", "./baseline_eval/stylegan2_15000kimgs.pt"),
    ("20000kimgs", "./baseline_eval/stylegan2_20000kimgs.pt"),
    ("25000kimgs", "./baseline_eval/stylegan2_final.pt"),
]

# same seed for fair comparison
torch.manual_seed(42)
z = torch.randn(16, Z_DIM, device=device)

for name, ckpt_path in CHECKPOINTS:
    generator = Generator(...)
    ckpt = torch.load(ckpt_path, ...)
    generator.load_state_dict(ckpt["ema"])
    generator.eval()
    generator.compute_w_mean(...)
    
    with torch.no_grad():
        imgs = generator(z, truncation_psi=0.7)
        imgs = (imgs.clamp(-1, 1) + 1) / 2
    vutils.save_image(imgs, f"progression_{name}.png", nrow=4, padding=2)