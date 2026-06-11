import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from collections import defaultdict
from typing import Optional
from utils.dataset import build_dataloader
 
from training.ada import ADA

def _infinite(loader: DataLoader):
    """Wraps a DataLoader to yield batches indefinitely."""
    while True:
        yield from loader


def _sample_z(batch_size: int, z_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, z_dim, device=device)


def _clip_grad(model: nn.Module, max_norm: float = 1.0):
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _save_checkpoint(path: str, *, generator, discriminator, ema, g_opt, d_opt, step, ada_p):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "generator":      generator.state_dict(),
        "discriminator":  discriminator.state_dict(),
        "ema":            ema.shadow.state_dict(),
        "g_opt":          g_opt.state_dict(),
        "d_opt":          d_opt.state_dict(),
        "step":           step,
        "ada_p":          ada_p,
    }, path)
    print(f"  ✓ Saved checkpoint → {path}")


def load_checkpoint(path: str, generator, discriminator, ema, g_opt, d_opt, ada):
    """Restores all components from a checkpoint saved by _save_checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    ema.shadow.load_state_dict(ckpt["ema"])
    g_opt.load_state_dict(ckpt["g_opt"])
    d_opt.load_state_dict(ckpt["d_opt"])
    ada.p = ckpt["ada_p"]
    return ckpt["step"]

def get_dropout_p(discriminator):
    for block in discriminator.blocks:
        if hasattr(block, 'dropout'):
            if hasattr(block.dropout, 'current_p'):
                return block.dropout.current_p  # adaptive
            else:
                return block.dropout.p  # fixed
    return 0.0

class GeneratorEMA:
    """Keeps a shadow copy of the generator with exponential moving average."""

    def __init__(self, generator: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(generator).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, generator: nn.Module):
        for s_param, param in zip(self.shadow.parameters(), generator.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)

from torch.utils.data import DataLoader
from copy import deepcopy
from collections import defaultdict
from typing import Optional
import argparse

def train(
    generator: nn.Module,
    discriminator: nn.Module,
    loss_fn,                          # your loss class instance
    dataloader: DataLoader,
    ada: ADA,
    *,
    # optimisers
    g_opt: torch.optim.Optimizer,
    d_opt: torch.optim.Optimizer,
    # schedule
    total_kimgs: float = 25000,       # total training kimgs
    batch_size: int = 32,
    z_dim: int = 512,
    # regularisation intervals (lazy regularisation)
    d_reg_every: int = 16,            # R1 every N steps
    g_reg_every: int = 4,             # path-length every N steps (0 = off)
    # EMA
    ema_decay: float = 0.9999,
    # checkpointing
    save_every_kimgs: float = 1000,
    ckpt_path: str = "/home/elicer/KU_DATA303_TEAM05/checkpoints/stylegan2",
    # misc
    device: str = "cuda",
    log_every: int = 100,             # steps
    start_step: int = 0,
):
    log_file = open(f"{ckpt_path}_log.txt", "a")
    device = torch.device(device)
    generator = generator.to(device).train()
    discriminator = discriminator.to(device).train()

    ema = GeneratorEMA(generator, decay=ema_decay)

    dataloader = build_dataloader(batch_size=batch_size, num_workers=2)
    print(f"Dataset loaded: {len(dataloader.dataset):,} images")

    total_steps = int(total_kimgs * 1000 / batch_size)
    kimgs_per_step = batch_size / 1000.0
    next_save_kimg = save_every_kimgs

    loader_iter = _infinite(dataloader)
    metrics = defaultdict(float)
    step = start_step

    print(f"Training for {total_kimgs:.0f} kimgs  ({total_steps} steps)")

    while step < total_steps:
        kimg = step * kimgs_per_step

        # ------------------------------------------------------------------ #
        # 1.  Discriminator step
        # ------------------------------------------------------------------ #
        real = next(loader_iter).to(device, non_blocking=True)
        real_aug = ada(real)

        z = _sample_z(batch_size, z_dim, device)
        with torch.no_grad():
            fake = generator(z)
        fake_aug = ada(fake.detach())

        d_opt.zero_grad(set_to_none=True)

        # forward pass through D to get logits
        real_logits = discriminator(real_aug)
        fake_logits = discriminator(fake_aug)

        d_loss = loss_fn.discriminator_loss(real_logits, fake_logits)
        d_loss.backward()

        # lazy R1 — needs a fresh forward with requires_grad so autograd can trace
        if step % d_reg_every == 0:
            real_r1 = real.detach().requires_grad_(True)   # unaugmented
            real_logits_r1 = discriminator(real_r1)
            r1 = loss_fn.r1_penalty(real_r1, real_logits_r1)
            (r1 * d_reg_every).backward()

        _clip_grad(discriminator)
        d_opt.step()

        # ADA update — use real_logits from the D forward pass above
        rt_sign = real_logits.detach().sign().float().mean().item()
        ada.update_p(rt_sign)

        progress = step / total_steps
        discriminator.set_dropout_p(progress)


        # ------------------------------------------------------------------ #
        # 2.  Generator step
        # ------------------------------------------------------------------ #
        z = _sample_z(batch_size, z_dim, device)
        fake = generator(z)
        fake_aug = ada(fake)   # augment before passing to D

        fake_logits = discriminator(fake_aug)
        g_loss = loss_fn.generator_loss(fake_logits, fake_images=fake_aug)

        g_opt.zero_grad(set_to_none=True)
        g_loss.backward()
        _clip_grad(generator)
        g_opt.step()
        ema.update(generator)

        # ------------------------------------------------------------------ #
        # 3.  Logging
        # ------------------------------------------------------------------ #
        # for k, v in {**d_dict, **g_dict}.items():
        #     metrics[k] += v.item() if torch.is_tensor(v) else float(v)

        if step % log_every == 0 and step > 0:
            dropout_p = get_dropout_p(discriminator)
            log_line = (
                f"step {step:7d}  kimg {kimg:8.1f}"
                f"  d_loss {d_loss.item():.4f}"
                f"  g_loss {g_loss.item():.4f}"
                f"  ada_p {ada.p:.3f}"
                f"  dropout_p {dropout_p:.3f}\n"
            )
            print(log_line, end="")
            log_file.write(log_line)
            if step % (log_every * 10) == 0:
                log_file.flush()
            metrics.clear()

        # temporary debug — remove after confirming channels are balanced
        if step % 200 == 0 and step <= 1000:
            with torch.no_grad():
                test_z = _sample_z(4, z_dim, device)
                test_img = generator(test_z)
                print(f"  R mean: {test_img[:,0].mean():.3f}  "
                    f"G mean: {test_img[:,1].mean():.3f}  "
                    f"B mean: {test_img[:,2].mean():.3f}")

        # ------------------------------------------------------------------ #
        # 4.  Checkpointing
        # ------------------------------------------------------------------ #
        if kimg >= next_save_kimg:
            _save_checkpoint(
                path=f"{ckpt_path}_{kimg:.0f}kimgs.pt",
                generator=generator,
                discriminator=discriminator,
                ema=ema,
                g_opt=g_opt,
                d_opt=d_opt,
                step=step,
                ada_p=ada.p,
            )
            next_save_kimg += save_every_kimgs

        step += 1

    # final checkpoint
    _save_checkpoint(
        path=f"{ckpt_path}_final.pt",
        generator=generator,
        discriminator=discriminator,
        ema=ema,
        g_opt=g_opt,
        d_opt=d_opt,
        step=step,
        ada_p=ada.p,
    )
    print("Training complete.")
    log_file.close()
    return ema