import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image

class FFHQDataset(Dataset):
    def __init__(
        self,
        root: str = "/home/elicer/ffhq_data/images",
        transform=None,
        max_samples=None,
    ):
        self.transform = transform
        self._ds = ImageFolder(root=root)

        if max_samples is not None:
            self._ds.samples = self._ds.samples[:max_samples]
            self._ds.targets = self._ds.targets[:max_samples]

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx):
        img, _ = self._ds[idx]  # ignore label
        if self.transform is not None:
            img = self.transform(img)
        return img

def build_dataloader(
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 64,
    initial_ada_p: float = 0.0,
    pin_memory: bool = True,
    streaming: bool = False,
    max_samples: int=2000,
    local_dir="/home/elicer/ffhq_data",
):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    ])
    images_path = download_dataset(local_dir)
    dataset = FFHQDataset(
        root=images_path,
        transform=transform,
        max_samples=max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not streaming),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    return loader

def download_dataset(local_dir="/home/elicer/ffhq_data"):
    os.makedirs(local_dir, exist_ok=True)
    
    zip_path = os.path.join(local_dir, "ffhq-64x64.zip")
    images_path = os.path.join(local_dir, "images")

    # skip if already downloaded
    if os.path.exists(images_path):
        print(f"Dataset already exists at {images_path}, skipping download.")
        return images_path

    print("Downloading FFHQ-64x64...")
    hf_hub_download(
        repo_id="Dmini/FFHQ-64x64",
        filename="ffhq-64x64.zip",
        repo_type="dataset",
        local_dir=local_dir,
    )

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(images_path)

    print(f"Dataset ready at {images_path}")
    return images_path