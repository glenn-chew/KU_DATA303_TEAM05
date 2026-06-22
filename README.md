# StyleGAN2-ADA Implementation (KU_DATA303_TEAM05)

Implementation of StyleGAN2 with Adaptive Discriminator Augmentation (ADA) for face generation on FFHQ-64x64.

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (minimum 8GB VRAM recommended)
- Tested on NVIDIA A100 80GB PCIe

### Software
```bash
# Option 1 — Conda (recommended)
conda env create -f environment.yml
conda activate stylegan2-ada

# Option 2 — Pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Dataset

The FFHQ-64x64 dataset (~878MB) will be downloaded automatically on first run.

To download manually:
```bash
python -c "from utils.dataset import download_dataset; download_dataset()"
```

## Quick Verification (Smoke Test)

Verify the implementation works without full training (~5 minutes on GPU):
```bash
python smoketest.py
```

## Training

### Basic training:
```bash
python train.py
```

### With custom settings:
```bash
python train.py \
  --total_kimgs 25000 \
  --batch_size 64 \
  --lr 2e-4 \
  --ada_target 0.6 \
  --ada_kimg 100
```

### Resume from checkpoint:
```bash
python train.py --resume ./checkpoints/stylegan2_final.pt --total_kimgs {checkpoint_number}
```

### Ablation experiments:
```bash
# No ADA
python train.py --ada_kimg 99999999 --max_samples 2000

# Standard ADA
python train.py --ada_kimg 100 --max_samples 2000

# ADA + Fixed Dropout
python train.py --ada_kimg 100 --max_samples 2000 --dropout_p 0.3

# ADA + Progressive Dropout
python train.py --ada_kimg 100 --max_samples 2000 --dropout_p 0.3 --adaptive true
```

## Generate Samples 
This code generates samples from the final checkpoint obtained when the training completes.
Edit the ckpt_path to relevant checkpoint to generate samples from intermediate checkpoints.
```bash
python generate.py
```

## Compute FID Score
FID for the trained model is calculated, and the output is a csv file.
A complete training is required. Otherwise, manually edit the checkpoints.
```bash
python fid_eval.py
```

## Project Structure
```
KU_DATA303_TEAM05/
├── networks/
│   ├── generator.py      # Generator, MappingNetwork, SynthesisNetwork
│   ├── discriminator.py  # Discriminator, ResBlock, MinibatchStdDev
│   └── layers.py         # ModulatedConv2d, EqualLinear, StyleBlock, ToRGB
├── training/
│   ├── trainer.py        # Training loop, GeneratorEMA
│   ├── ada.py            # ADA augmentation pipeline
│   └── loss.py           # StyleGAN2Loss, R1 penalty
├── utils/
│   └── dataset.py        # FFHQDataset, DataLoader
├── train.py              # Entry point
├── smoketest.py          # Quick verification
├── generate.py           # Sample generation
├── fid_eval.py           # FID evaluation
├── show_augmentations.py # Display ADA augmentations
├── requirements.txt      # Pip dependencies
├── environment.yml       # Conda environment
└── README.md
```

## References

- Karras et al. "Training Generative Adversarial Networks with Limited Data" (NeurIPS 2020)
- Karras et al. "Analyzing and Improving the Image Quality of StyleGAN" (CVPR 2020)