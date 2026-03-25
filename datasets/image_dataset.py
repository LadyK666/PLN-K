from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class ImageOnlyDataset(Dataset):
    """
    Minimal image dataset for scaffolding (no labels yet).

    It scans image files under:
      root_dir / split_dir  (e.g. split_dir='train')

    and returns a dict with:
      - 'image': Tensor(C,H,W)
      - 'path': str
    """

    def __init__(
        self,
        root_dir: str,
        split_dir: str = "train",
        transform: Optional[Callable] = None,
        exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ):
        self.root_dir = Path(root_dir)
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.exts = tuple(e.lower() for e in exts)

        data_dir = self.root_dir / self.split_dir
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset split dir not found: {data_dir}")

        self.image_paths = sorted(
            [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in self.exts]
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found under: {data_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img_t = self.transform(img)
        else:
            # Fallback: float tensor in [0,1]
            img_t = TF.to_tensor(img)

        return {"image": img_t, "path": str(path)}

