import torch
from utils.dataset import build_dataloader
from training.trainer import GeneratorEMA, train
from training.ada import ADA

from networks.generator import Generator
from networks.discriminator import Discriminator
from training.loss import StyleGAN2Loss

device = "cuda" if torch.cuda.is_available() else "cpu"

# tiny hyperparams
BATCH_SIZE = 4
Z_DIM = 512
IMAGE_SIZE = 64

# data — just grab one batch
loader = build_dataloader(batch_size=BATCH_SIZE, num_workers=0)
real = next(iter(loader)).to(device)
print(f"✓ Data loading OK — batch shape: {real.shape}")  # expect (4, 3, 64, 64)
print(f"  pixel range: [{real.min():.2f}, {real.max():.2f}]")  # expect [-1, 1]

# models
generator     = Generator(z_dim=Z_DIM, img_resolution=IMAGE_SIZE).to(device)
discriminator = Discriminator(img_resolution=IMAGE_SIZE).to(device)

# forward passes
z = torch.randn(BATCH_SIZE, Z_DIM, device=device)
fake = generator(z)
print(f"✓ Generator OK — output shape: {fake.shape}")  # expect (4, 3, 64, 64)

real_logits = discriminator(real)
fake_logits = discriminator(fake.detach())
print(f"✓ Discriminator OK — logits shape: {real_logits.shape}")  # expect (4, 1) or (4,)

# loss
loss_fn = StyleGAN2Loss(r1_gamma=10.0)
d_loss = loss_fn.discriminator_loss(real_logits, fake_logits)
g_loss = loss_fn.generator_loss(fake_logits)
print(f"✓ Losses OK — d_loss: {d_loss.item():.4f}  g_loss: {g_loss.item():.4f}")

# R1 penalty
real_r1 = real.detach().requires_grad_(True)
real_logits_r1 = discriminator(real_r1)
r1 = loss_fn.r1_penalty(real_r1, real_logits_r1)
print(f"✓ R1 penalty OK — r1: {r1.item():.4f}")

# ADA
ada = ADA(augment_p=0.5)
real_aug = ada(real)
print(f"✓ ADA OK — augmented shape: {real_aug.shape}")

# backward passes
d_loss.backward()
print("✓ D backward OK")

g_loss = loss_fn.generator_loss(discriminator(ada(generator(z))))
g_loss.backward()
print("✓ G backward OK")

# EMA
ema = GeneratorEMA(generator)
ema.update(generator)
print("✓ EMA OK")

# mini training run — 10 steps only
print("\nRunning 10 steps...")
ada_mini = ADA(augment_p=0.0)
loader_mini = build_dataloader(batch_size=BATCH_SIZE, num_workers=0)

g_opt = torch.optim.Adam(generator.parameters(), lr=2e-3, betas=(0.0, 0.99))
d_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-3, betas=(0.0, 0.99))

# add temporarily to smoketest
real = next(iter(loader)).to(device)
z = torch.randn(BATCH_SIZE, Z_DIM, device=device)
fake = generator(z)

print(f"real range: [{real.min():.3f}, {real.max():.3f}]")
print(f"fake range: [{fake.min():.3f}, {fake.max():.3f}]")
print(f"fake mean: {fake.mean():.3f}, std: {fake.std():.3f}")

train(
    generator=generator,
    discriminator=discriminator,
    loss_fn=loss_fn,
    dataloader=loader_mini,
    ada=ada_mini,
    g_opt=g_opt,
    d_opt=d_opt,
    total_kimgs=BATCH_SIZE * 10 / 1000,  # exactly 10 steps
    batch_size=BATCH_SIZE,
    z_dim=Z_DIM,
    device=device,
    log_every=5,
    save_every_kimgs=999,   # never saves during smoke test
)
print("✓ Full training loop OK")

