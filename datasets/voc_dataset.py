from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from torchvision.transforms import functional as TF

from .voc_transforms import RandomChoiceSSDOrModifiedYOLO  # type: ignore


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def _parse_voc_xml(xml_path: Path) -> Tuple[List[List[float]], List[int]]:
    """
    Parse VOC XML:
    - objects -> name + bndbox (xmin, ymin, xmax, ymax)
    - ignore parts and other sub-elements.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    boxes: List[List[float]] = []
    labels: List[int] = []

    name_to_idx = {n: i for i, n in enumerate(VOC_CLASSES)}

    for obj in root.findall("object"):
        name_node = obj.find("name")
        bndbox_node = obj.find("bndbox")
        if name_node is None or bndbox_node is None:
            continue
        cls_name = (name_node.text or "").strip()
        if cls_name not in name_to_idx:
            continue

        def _get_float(tag: str) -> float:
            v = bndbox_node.find(tag)
            if v is None or v.text is None:
                return 0.0
            try:
                return float(v.text)
            except ValueError:
                return 0.0

        xmin = _get_float("xmin")
        ymin = _get_float("ymin")
        xmax = _get_float("xmax")
        ymax = _get_float("ymax")

        # Basic validity filter.
        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(name_to_idx[cls_name])

    return boxes, labels


@dataclass
class VOC2007DatasetConfig:
    voc_root_dir: str
    split: str = "trainval"
    output_size: Tuple[int, int] = (300, 300)  # (H, W)
    augment: bool = True


class VOC2007Dataset(Dataset):
    """
    VOC2007 dataset loader for detection tasks.

    Expected directory layout (VOCdevkit standard):
      {voc_root_dir}/Annotations/*.xml
      {voc_root_dir}/JPEGImages/*.jpg
      {voc_root_dir}/ImageSets/Main/{split}.txt

    Example voc_root_dir:
      /path/to/VOCdevkit/VOC2007
    """

    def __init__(
        self,
        voc_root_dir: str,
        split: str = "trainval",
        output_size: Tuple[int, int] = (300, 300),
        augment: bool = True,
        dataset_tag: str = "",
        normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.voc_root_dir = Path(voc_root_dir)
        self.split = split
        self.output_size = output_size
        self.dataset_tag = dataset_tag

        self.ann_dir = self.voc_root_dir / "Annotations"
        self.img_dir = self.voc_root_dir / "JPEGImages"
        self.imgset_dir = self.voc_root_dir / "ImageSets" / "Main"
        self.imgset_file = self.imgset_dir / f"{split}.txt"

        if not self.imgset_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.imgset_file}")

        with open(self.imgset_file, "r") as f:
            ids = [line.strip() for line in f.readlines() if line.strip()]
        self.ids = ids

        self.aug = RandomChoiceSSDOrModifiedYOLO(output_size=output_size) if augment else None
        self.normalize = Normalize(mean=list(normalize_mean), std=list(normalize_std))

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, img_id: str) -> Image.Image:
        # VOC format uses "{id}.jpg"
        img_path = self.img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")

    def _load_target(self, img_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        xml_path = self.ann_dir / f"{img_id}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Annotation not found: {xml_path}")

        boxes, labels = _parse_voc_xml(xml_path)
        if len(boxes) == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        image = self._load_image(img_id)
        boxes, labels = self._load_target(img_id)

        # Apply augmentation (geometric + color jitter). Output boxes are scaled to output_size.
        if self.aug is not None:
            image, boxes, labels = self.aug(image, boxes, labels)
        else:
            out_h, out_w = self.output_size
            w, h = image.size
            image = TF.resize(image, [out_h, out_w], interpolation=TF.InterpolationMode.BILINEAR)
            sx = out_w / float(w)
            sy = out_h / float(h)
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] *= sx
                boxes[:, [1, 3]] *= sy

        # To tensor + normalize
        image_t = TF.to_tensor(image)  # [0,1], shape (3,H,W)
        image_t = self.normalize(image_t)

        target = {
            "boxes": boxes,  # (N,4) in pixel coords after resize
            "labels": labels,  # (N,)
            "image_id": f"{self.dataset_tag}_{img_id}" if self.dataset_tag else img_id,
        }

        return {"image": image_t, "target": target}

