# compute_fid_all.py
import os
import torch
import torchvision.utils as vutils
from cleanfid import fid
from networks.generator import Generator
import csv

device = torch.device("cuda")
Z_DIM = 512
IMAGE_SIZE = 64
REAL_DIR = "../ffhq_data/images"
FAKE_BASE = "./fid_samples"
CSV_FILE = "./fid_results.csv"


MODELS = {
    "no_ada": [
        ("5000",  ".../noada/stylegan2_5000kimgs.pt"),
        ("10000", ".../noada/stylegan2_10000kimgs.pt"),
        ("15000", ".../noada/stylegan2_15000kimgs.pt"),
        ("20000", ".../noada/stylegan2_20000kimgs.pt"),
        ("25000", ".../noada/stylegan2_final.pt"),
    ],
    "std_ada": [
        ("5000",  ".../standardada/stylegan2_5000kimgs.pt"),
        ("10000", ".../standardada/stylegan2_10000kimgs.pt"),
        ("15000", ".../standardada/stylegan2_15000kimgs.pt"),
        ("20000", ".../standardada/stylegan2_20000kimgs.pt"),
        ("25000", ".../standardada/stylegan2_final.pt"),
    ],
    "dropout": [
        ("5000",  ".../dropout/stylegan2_5000kimgs.pt"),
        ("15000", ".../dropout/stylegan2_15000kimgs.pt"),
        ("20000", ".../dropout/stylegan2_20000kimgs.pt"),
        ("25000", ".../dropout/stylegan2_final.pt"),
    ],
}

results = {}

for model_name, ckpt_path in MODELS.items():
    print(f"\nEvaluating {model_name}...")

    # load generator
    generator = Generator(z_dim=Z_DIM, img_resolution=IMAGE_SIZE).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["ema"])
    generator.eval()
    generator.compute_w_mean(n_samples=10000, device=device)

    # generate 10k fake images
    fake_dir = os.path.join(FAKE_BASE, model_name)
    os.makedirs(fake_dir, exist_ok=True)

    print(f"Generating 10,000 images for {model_name}...")
    with torch.no_grad():
        for i in range(10000 // 64):
            z = torch.randn(64, Z_DIM, device=device)
            imgs = generator(z, truncation_psi=1.0)
            imgs = (imgs.clamp(-1, 1) + 1) / 2
            for j, img in enumerate(imgs):
                vutils.save_image(img, f"{fake_dir}/{i*64+j:05d}.png")

    # compute FID
    score = fid.compute_fid(REAL_DIR, fake_dir, mode="clean", device=device)
    results[model_name] = score
    print(f"FID {model_name}: {score:.2f}")


# save as CSV
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "kimgs", "fid"])  # header
    for model_name, kimg_scores in results.items():
        for kimg, score in kimg_scores.items():
            writer.writerow([model_name, kimg, score])

print(f"✓ Saved {CSV_FILE}")