from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


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
CLS2ID = {c: i for i, c in enumerate(VOC_CLASSES)}


def read_ids(voc_root: Path, split: str) -> List[str]:
    p = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]


def read_boxes(voc_root: Path, img_id: str) -> Tuple[np.ndarray, np.ndarray]:
    xml_path = voc_root / "Annotations" / f"{img_id}.xml"
    root = ET.parse(xml_path).getroot()
    boxes = []
    labels = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        bb = obj.find("bndbox")
        if name_node is None or bb is None:
            continue
        cls_name = (name_node.text or "").strip()
        if cls_name not in CLS2ID:
            continue
        x1 = float(bb.find("xmin").text)
        y1 = float(bb.find("ymin").text)
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(CLS2ID[cls_name])
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def iou_upper_triangle(boxes: np.ndarray) -> np.ndarray:
    n = boxes.shape[0]
    if n < 2:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = inter / (area[:, None] + area[None, :] - inter + 1e-7)
    iu = np.triu_indices(n, 1)
    return iou[iu]


def summarize(voc_root: Path, split: str) -> Dict:
    ids = read_ids(voc_root, split)
    per_img_count: List[int] = []
    iou_all: List[float] = []
    iou_same: List[float] = []
    area_all: List[float] = []

    for img_id in ids:
        boxes, labels = read_boxes(voc_root, img_id)
        n = boxes.shape[0]
        per_img_count.append(n)
        if n == 0:
            continue

        area_all.extend(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).tolist())
        if n >= 2:
            iou_all.extend(iou_upper_triangle(boxes).tolist())
            for c in np.unique(labels):
                idx = np.where(labels == c)[0]
                if idx.size >= 2:
                    iou_same.extend(iou_upper_triangle(boxes[idx]).tolist())

    per_img = np.asarray(per_img_count, dtype=np.float32)
    area_arr = np.asarray(area_all, dtype=np.float32) if len(area_all) else np.asarray([0.0], dtype=np.float32)
    iou_all_arr = np.asarray(iou_all, dtype=np.float32) if len(iou_all) else np.asarray([0.0], dtype=np.float32)
    iou_same_arr = np.asarray(iou_same, dtype=np.float32) if len(iou_same) else np.asarray([0.0], dtype=np.float32)

    return {
        "num_images": int(len(ids)),
        "objects_per_image_mean": float(per_img.mean()) if per_img.size else 0.0,
        "objects_per_image_q50_q90_q95_max": [
            float(np.quantile(per_img, 0.50)),
            float(np.quantile(per_img, 0.90)),
            float(np.quantile(per_img, 0.95)),
            int(per_img.max()) if per_img.size else 0,
        ],
        "iou_all_q50_q90_q95": [
            float(np.quantile(iou_all_arr, 0.50)),
            float(np.quantile(iou_all_arr, 0.90)),
            float(np.quantile(iou_all_arr, 0.95)),
        ],
        "iou_same_q50_q90_q95": [
            float(np.quantile(iou_same_arr, 0.50)),
            float(np.quantile(iou_same_arr, 0.90)),
            float(np.quantile(iou_same_arr, 0.95)),
        ],
        "box_area_q10_q50_q90": [
            float(np.quantile(area_arr, 0.10)),
            float(np.quantile(area_arr, 0.50)),
            float(np.quantile(area_arr, 0.90)),
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser("NMS/IoU stats helper for VOC")
    p.add_argument(
        "--voc2007_root",
        type=str,
        default="",
        help="Path to VOCdevkit/VOC2007 (e.g. /path/to/VOCdevkit/VOC2007). Required.",
    )
    p.add_argument(
        "--voc2012_root",
        type=str,
        default="",
        help="Path to VOCdevkit/VOC2012 (e.g. /path/to/VOCdevkit/VOC2012). Optional.",
    )
    p.add_argument("--split2007", type=str, default="test", choices=["trainval", "train", "val", "test"])
    p.add_argument("--use_voc2012", action="store_true", help="Also summarize VOC2012 trainval (requires --voc2012_root).")
    args = p.parse_args()

    if not args.voc2007_root:
        raise ValueError("--voc2007_root is required (no machine-specific default is bundled).")

    root2007 = Path(args.voc2007_root)
    print(f"VOC2007 {args.split2007}:", summarize(root2007, args.split2007))
    print("VOC2007 trainval:", summarize(root2007, "trainval"))

    if args.use_voc2012:
        if not args.voc2012_root:
            raise ValueError("--voc2012_root is required when --use_voc2012 is set.")
        root2012 = Path(args.voc2012_root)
        print("VOC2012 trainval:", summarize(root2012, "trainval"))


if __name__ == "__main__":
    main()

