from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_images(batch: List[Dict[str, Any]]):
    """
    Collate for image-only dataset.

    Assumes all images in the batch are already the same spatial size.
    """
    images = [b["image"] for b in batch]
    paths = [b["path"] for b in batch]
    return {"image": torch.stack(images, dim=0), "path": paths}

