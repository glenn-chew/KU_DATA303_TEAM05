# add this to generate.py or a separate fid_generate.py
import os
import torch
import torchvision.utils as vutils
from networks.generator import Generator
from cleanfid import fid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z_DIM = 512
IMAGE_SIZE = 64
# CKPT_PATH = "/home/elicer/KU_DATA303_TEAM05/checkpoints/stylegan2_final.pt"
# FAKE_DIR = "/home/elicer/KU_DATA303_TEAM05/fid_samples/fake"
CKPT_PATH = "/checkpoints/stylegan2_final.pt"
FAKE_DIR = "/fid_samples/fake"


os.makedirs(FAKE_DIR, exist_ok=True)

generator = Generator(z_dim=Z_DIM, img_resolution=IMAGE_SIZE).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
generator.load_state_dict(ckpt["ema"])
generator.eval()
generator.compute_w_mean(n_samples=10000, device=device)

# generate 10k images
print("Generating 10,000 images...")
batch_size = 64
n_batches = 10000 // batch_size

with torch.no_grad():
    for i in range(n_batches):
        z = torch.randn(batch_size, Z_DIM, device=device)
        imgs = generator(z, truncation_psi=1.0)  # use psi=1.0 for FID
        imgs = (imgs.clamp(-1, 1) + 1) / 2  # → [0, 1]
        for j, img in enumerate(imgs):
            idx = i * batch_size + j
            vutils.save_image(img, f"{FAKE_DIR}/{idx:05d}.png")
        if i % 10 == 0:
            print(f"  {i * batch_size}/{10000} images generated")

print("Done generating images")

score = fid.compute_fid(
    "/home/elicer/ffhq_data/images/",  # real images folder
    "/home/elicer/KU_DATA303_TEAM05/fid_samples/fake",  # fake images folder
    mode="clean",
    device=device,
)
print(f"FID: {score:.2f}")