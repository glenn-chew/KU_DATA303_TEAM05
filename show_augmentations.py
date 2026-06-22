# visualize_augmentations.py
import torch
import torchvision.utils as vutils
import os
from utils.dataset import build_dataloader
from training.ada import ADA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./augmentations/augmentation_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load a batch of real images
loader = build_dataloader(batch_size=8, num_workers=0)
real = next(iter(loader)).to(device)
real_01 = (real.clamp(-1, 1) + 1) / 2

# save original
vutils.save_image(real_01, f"{OUTPUT_DIR}/original.png", nrow=8, padding=2)
print("✓ Saved original.png")

# visualize at different p values
for p in [0.25, 0.5, 0.75, 1.0]:
    ada = ADA(augment_p=p, use_color=True, use_geometric=True, use_blit=True)
    augmented = ada(real)
    augmented_01 = (augmented.clamp(-1, 1) + 1) / 2
    vutils.save_image(augmented_01, f"{OUTPUT_DIR}/augmented_p{p}.png", nrow=8, padding=2)
    print(f"✓ Saved augmented_p{p}.png")

# visualize individual augmentation types
for name, kwargs in [
    ("color_only",     {"use_color": True,  "use_geometric": False, "use_blit": False}),
    ("geometric_only", {"use_color": False, "use_geometric": True,  "use_blit": False}),
    ("blit_only", {"use_color": False, "use_geometric": False, "use_blit": True})
]:
    ada = ADA(augment_p=1.0, **kwargs)
    augmented = ada(real)
    augmented_01 = (augmented.clamp(-1, 1) + 1) / 2
    vutils.save_image(augmented_01, f"{OUTPUT_DIR}/{name}.png", nrow=8, padding=2)
    print(f"✓ Saved {name}.png")

# visualize frequency augmentation separately
# add this after you implement it
# ada_freq = ADA(augment_p=1.0, use_frequency=True)
# augmented = ada_freq(real)
# augmented_01 = (augmented.clamp(-1, 1) + 1) / 2
# vutils.save_image(augmented_01, f"{OUTPUT_DIR}/frequency_aug.png", nrow=8, padding=2)