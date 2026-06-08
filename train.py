import argparse
import torch

from utils.dataset import build_dataloader
from training.trainer import train, load_checkpoint, GeneratorEMA
from training.ada import ADA

from networks.generator import Generator
from networks.discriminator import Discriminator
from training.loss import StyleGAN2Loss

def parse_args():
    p = argparse.ArgumentParser(description="StyleGAN2-ADA · FFHQ-64x64")

    # Data
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--image_size",      type=int,   default=64)

    # Architecture
    p.add_argument("--z_dim",           type=int,   default=512)
    p.add_argument("--dropout_p", type=float, default=0.0)

    # Training schedule
    p.add_argument("--total_kimgs",     type=float, default=1000)
    p.add_argument("--d_reg_every",     type=int,   default=16,
                   help="R1 regularisation interval (lazy reg)")
    p.add_argument("--g_reg_every",     type=int,   default=4,
                   help="Path-length regularisation interval (0 = off)")

    # Optimiser
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--beta1",           type=float, default=0.0)
    p.add_argument("--beta2",           type=float, default=0.99)

    # EMA
    p.add_argument("--ema_decay",       type=float, default=0.9999)

    # ADA
    p.add_argument("--ada_target",      type=float, default=0.6)
    p.add_argument("--ada_kimg",        type=float, default=100.0,
                   help="Kimgs over which ADA ramps p by 1 (higher = slower)")
    p.add_argument("--ada_interval",    type=int,   default=4)
    p.add_argument('--use_ada', type=lambda x: x.lower() == 'true', default=True)

    # Checkpointing
    p.add_argument("--save_every_kimgs", type=float, default=1000)
    p.add_argument("--ckpt_path",       type=str,   default="/home/elicer/KU_DATA303_TEAM05/checkpoints/stylegan2")
    p.add_argument("--resume",          type=str,   default=None,
                   help="Path to checkpoint to resume from")

    # Misc
    p.add_argument("--device",          type=str,   default="cuda")
    p.add_argument("--log_every",       type=int,   default=200)
    p.add_argument("--seed",            type=int,   default=42)

    # return p.parse_args()
    args, _ = p.parse_known_args()
    return args

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    dataloader = build_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        max_samples=2000,
    )
    print(f"Dataset loaded: {len(dataloader.dataset):,} images")

    # ------------------------------------------------------------------ #
    # Models 
    # ------------------------------------------------------------------ #
    generator     = Generator(z_dim=args.z_dim, img_resolution=args.image_size).to(device)
    discriminator = Discriminator(img_resolution=args.image_size,channel_base=8192, channel_max=256, dropout_p=args.dropout_p).to(device)

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #
    loss_fn       = StyleGAN2Loss(r1_gamma=10.0)
    ada = ADA(
        augment_p=0.0,
        ada_target=args.ada_target,
        ada_kimg=args.ada_kimg,
        ada_interval=args.ada_interval,
        use_ada=args.use_ada,
    )
    print(f"{args.use_ada}\n")

    # ------------------------------------------------------------------ #
    # Optimisers
    # Lazy-reg learning rate scaling: lr * (reg_interval / (reg_interval + 1))
    # ------------------------------------------------------------------ #
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) if args.g_reg_every > 0 else 1.0

    g_opt = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(args.beta1 ** g_reg_ratio, args.beta2 ** g_reg_ratio),
    )
    d_opt = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(args.beta1 ** d_reg_ratio, args.beta2 ** d_reg_ratio),
    )

    # ------------------------------------------------------------------ #
    # (Optional) Resume
    # ------------------------------------------------------------------ #
    start_step = 0
    if args.resume:
        ema_tmp = GeneratorEMA(generator, decay=args.ema_decay)
        start_step = load_checkpoint(
            args.resume, generator, discriminator, ema_tmp, g_opt, d_opt, ada
        )
    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    train(
        generator=generator,
        discriminator=discriminator,
        loss_fn=loss_fn,
        dataloader=dataloader,
        ada=ada,              
        g_opt=g_opt,
        d_opt=d_opt,
        total_kimgs=args.total_kimgs,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        d_reg_every=args.d_reg_every,
        g_reg_every=0,        
        ema_decay=args.ema_decay,
        save_every_kimgs=args.save_every_kimgs,
        ckpt_path="/home/elicer/KU_DATA303_TEAM05/checkpoints/stylegan2",
        device=str(device),
        log_every=args.log_every,
        start_step=start_step,
    )


if __name__ == "__main__":
    main()