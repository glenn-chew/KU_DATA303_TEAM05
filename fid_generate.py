# generate_all.py
import os
import torch
import torchvision.utils as vutils
from networks.generator import Generator

# config
Z_DIM = 512
IMAGE_SIZE = 64
OUTPUT_DIR = "/home/elicer/KU_DATA303_TEAM05/samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------ #
# 1. Progression grids (baseline only)
# ------------------------------------------------------------------ #
CHECKPOINTS = [
    ("1000",  "/home/elicer/KU_DATA303_TEAM05/baseline_eval/stylegan2_1000kimgs.pt"),
    ("5000",  "/home/elicer/KU_DATA303_TEAM05/baseline_eval/stylegan2_5000kimgs.pt"),
    ("10000", "/home/elicer/KU_DATA303_TEAM05/baseline_eval/stylegan2_10000kimgs.pt"),
    ("15000", "/home/elicer/KU_DATA303_TEAM05/baseline_eval/stylegan2_15000kimgs.pt"),
    ("20000", "/home/elicer/KU_DATA303_TEAM05/baseline_eval/stylegan2_20000kimgs.pt"),
    ("25000", "/home/elicer/KU_DATA303_TEAM05/baseline_eval/stylegan2_final.pt"),
]

torch.manual_seed(42)
z = torch.randn(16, Z_DIM, device=device)

for name, ckpt_path in CHECKPOINTS:
    generator = Generator(z_dim=Z_DIM, img_resolution=IMAGE_SIZE).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["ema"])
    generator.eval()
    generator.compute_w_mean(n_samples=10000, device=device)

    with torch.no_grad():
        imgs = generator(z, truncation_psi=0.7)
        imgs = (imgs.clamp(-1, 1) + 1) / 2
    vutils.save_image(imgs, f"{OUTPUT_DIR}/progression_{name}kimgs.png", nrow=4, padding=2)
    print(f"✓ Saved progression_{name}kimgs.png")

# ------------------------------------------------------------------ #
# 2. Truncation comparison (baseline final)
# ------------------------------------------------------------------ #
generator = Generator(z_dim=Z_DIM, img_resolution=IMAGE_SIZE).to(device)
ckpt = torch.load(CHECKPOINTS[-1][1], map_location=device, weights_only=False)
generator.load_state_dict(ckpt["ema"])
generator.eval()
generator.compute_w_mean(n_samples=10000, device=device)

torch.manual_seed(42)
z = torch.randn(8, Z_DIM, device=device)

all_imgs = []
for psi in [0.5, 0.7, 1.0]:
    with torch.no_grad():
        imgs = generator(z, truncation_psi=psi)
        imgs = (imgs.clamp(-1, 1) + 1) / 2
    all_imgs.append(imgs)

combined = torch.cat(all_imgs, dim=0)
vutils.save_image(combined, f"{OUTPUT_DIR}/truncation_comparison.png", nrow=8, padding=2)
print("✓ Saved truncation_comparison.png")

# ------------------------------------------------------------------ #
# 3. Ablation final grids (no_ada, std_ada, dropout)
# ------------------------------------------------------------------ #
ABLATION = [
    ("noada",    "/home/elicer/KU_DATA303_TEAM05/noada_eval/stylegan2_final.pt"),
    ("stdada",   "/home/elicer/KU_DATA303_TEAM05/baseline_eval/stylegan2_final.pt"),
    ("dropout",  "/home/elicer/KU_DATA303_TEAM05/dropout_eval/stylegan2_final.pt"),
]

torch.manual_seed(42)
z = torch.randn(16, Z_DIM, device=device)

for name, ckpt_path in ABLATION:
    generator = Generator(z_dim=Z_DIM, img_resolution=IMAGE_SIZE).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["ema"])
    generator.eval()
    generator.compute_w_mean(n_samples=10000, device=device)

    with torch.no_grad():
        imgs = generator(z, truncation_psi=0.7)
        imgs = (imgs.clamp(-1, 1) + 1) / 2
    vutils.save_image(imgs, f"{OUTPUT_DIR}/ablation_{name}.png", nrow=4, padding=2)
    print(f"✓ Saved ablation_{name}.png")

# ------------------------------------------------------------------ #
# 4. Interpolation (baseline final)
# ------------------------------------------------------------------ #
generator = Generator(z_dim=Z_DIM, img_resolution=IMAGE_SIZE).to(device)
ckpt = torch.load(CHECKPOINTS[-1][1], map_location=device, weights_only=False)
generator.load_state_dict(ckpt["ema"])
generator.eval()
generator.compute_w_mean(n_samples=10000, device=device)

torch.manual_seed(42)
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

print("\nAll done!")