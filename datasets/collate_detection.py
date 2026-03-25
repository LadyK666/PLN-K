from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


def collate_voc_detection(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate for detection:
    - images have same (H,W) due to fixed resize in dataset transforms
    - targets are variable-length lists of boxes/labels
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = [b["target"] for b in batch]
    return {"images": images, "targets": targets}

