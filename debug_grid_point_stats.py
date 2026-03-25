from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


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


def _read_split_ids(voc_root: Path, split: str) -> List[str]:
    p = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]


def _parse_boxes_and_size(voc_root: Path, img_id: str):
    xml_path = voc_root / "Annotations" / f"{img_id}.xml"
    root = ET.parse(xml_path).getroot()

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"Missing <size> in {xml_path}")
    w = float(size_node.find("width").text)
    h = float(size_node.find("height").text)

    boxes = []
    labels = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        bb = obj.find("bndbox")
        if name_node is None or bb is None:
            continue
        cls = (name_node.text or "").strip()
        if cls not in VOC_CLASSES:
            continue
        x1 = float(bb.find("xmin").text)
        y1 = float(bb.find("ymin").text)
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
        labels.append(cls)
    return w, h, boxes, labels


def _to_grid_xy(
    x: float,
    y: float,
    *,
    src_w: float,
    src_h: float,
    image_size: int,
    grid_size: int,
) -> Tuple[int, int]:
    # map to resized image coordinates first
    x_r = x / src_w * image_size
    y_r = y / src_h * image_size
    stride = image_size / grid_size
    u = int(x_r // stride)
    v = int(y_r // stride)
    u = max(0, min(grid_size - 1, u))
    v = max(0, min(grid_size - 1, v))
    return u, v


def main() -> None:
    p = argparse.ArgumentParser("Grid-point stats for PLN target construction")
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
        help="Path to VOCdevkit/VOC2012 (e.g. /path/to/VOCdevkit/VOC2012). Used when --use_voc2012.",
    )
    p.add_argument("--use_voc2012", action="store_true", help="Include VOC2012 split")
    p.add_argument("--split", type=str, default="trainval")
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--grid_size", type=int, default=14)
    p.add_argument("--B_point", type=int, default=2, help="B in your formula (2B is total-point cap)")
    p.add_argument("--output_json", type=str, default="")
    args = p.parse_args()

    if not args.voc2007_root:
        raise ValueError("--voc2007_root is required (no machine-specific default is bundled).")
    voc_roots = [Path(args.voc2007_root)]
    if args.use_voc2012:
        if not args.voc2012_root:
            raise ValueError("--voc2012_root is required when --use_voc2012 is set.")
        voc_roots.append(Path(args.voc2012_root))

    # Global stats
    total_images = 0
    total_boxes = 0
    total_points = 0
    total_center_points = 0
    total_corner_points = 0

    total_grid_cells_seen = 0
    grid_over_total = 0
    grid_over_center = 0
    grid_over_corner = 0

    class_counter_all_points = Counter()
    class_counter_overflow_cells = Counter()
    class_counter_multiclass_overflow_cells = Counter()

    # Per-image optional diagnostics (compact)
    img_overflow_examples = []
    overflow_image_count = 0
    overflow_cell_count = 0
    overflow_cell_single_class = 0
    overflow_cell_multi_class = 0

    max_total = 2 * args.B_point
    max_center = args.B_point
    max_corner = args.B_point

    for root in voc_roots:
        ids = _read_split_ids(root, args.split)
        for img_id in ids:
            total_images += 1
            w, h, boxes, labels = _parse_boxes_and_size(root, img_id)
            total_boxes += len(boxes)

            # Grid cell accumulators
            # key: (u,v) -> counts + class hist
            cell_total = defaultdict(int)
            cell_center = defaultdict(int)
            cell_corner = defaultdict(int)
            cell_cls_counter = defaultdict(Counter)

            for (x1, y1, x2, y2), cls in zip(boxes, labels):
                # center
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                u_c, v_c = _to_grid_xy(
                    cx, cy,
                    src_w=w, src_h=h,
                    image_size=args.image_size, grid_size=args.grid_size
                )
                key_c = (u_c, v_c)
                cell_total[key_c] += 1
                cell_center[key_c] += 1
                cell_cls_counter[key_c][cls] += 1

                total_points += 1
                total_center_points += 1
                class_counter_all_points[cls] += 1

                # four corners
                corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                for xx, yy in corners:
                    u_k, v_k = _to_grid_xy(
                        xx, yy,
                        src_w=w, src_h=h,
                        image_size=args.image_size, grid_size=args.grid_size
                    )
                    key_k = (u_k, v_k)
                    cell_total[key_k] += 1
                    cell_corner[key_k] += 1
                    cell_cls_counter[key_k][cls] += 1

                    total_points += 1
                    total_corner_points += 1
                    class_counter_all_points[cls] += 1

            # Evaluate per-cell thresholds for this image
            image_overflow = 0
            for key in set(list(cell_total.keys()) + list(cell_center.keys()) + list(cell_corner.keys())):
                total_grid_cells_seen += 1
                t = cell_total[key]
                c = cell_center[key]
                k = cell_corner[key]

                overflow_here = False
                if t > max_total:
                    grid_over_total += 1
                    overflow_here = True
                if c > max_center:
                    grid_over_center += 1
                    overflow_here = True
                if k > max_corner:
                    grid_over_corner += 1
                    overflow_here = True

                if overflow_here:
                    image_overflow += 1
                    overflow_cell_count += 1
                    class_counter_overflow_cells.update(cell_cls_counter[key])
                    num_cls = len(cell_cls_counter[key])
                    if num_cls <= 1:
                        overflow_cell_single_class += 1
                    else:
                        overflow_cell_multi_class += 1
                        class_counter_multiclass_overflow_cells.update(cell_cls_counter[key])

            if image_overflow > 0 and len(img_overflow_examples) < 20:
                img_overflow_examples.append(
                    {"dataset": root.name, "image_id": img_id, "overflow_cells": image_overflow}
                )
            if image_overflow > 0:
                overflow_image_count += 1

    single_ratio = (overflow_cell_single_class / overflow_cell_count) if overflow_cell_count > 0 else 0.0
    multi_ratio = (overflow_cell_multi_class / overflow_cell_count) if overflow_cell_count > 0 else 0.0
    overflow_image_rate = (overflow_image_count / total_images) if total_images > 0 else 0.0

    multi_overflow_class_ratio = []
    for cls, cnt_in_multi_over in class_counter_multiclass_overflow_cells.items():
        denom = class_counter_all_points.get(cls, 0)
        ratio = (cnt_in_multi_over / denom) if denom > 0 else 0.0
        multi_overflow_class_ratio.append((cls, ratio, cnt_in_multi_over, denom))
    multi_overflow_class_ratio.sort(key=lambda x: x[1], reverse=True)

    result = {
        "config": {
            "image_size": args.image_size,
            "grid_size": args.grid_size,
            "B_point": args.B_point,
            "max_total_points_per_cell": max_total,
            "max_center_points_per_cell": max_center,
            "max_corner_points_per_cell": max_corner,
            "split": args.split,
            "use_voc2012": args.use_voc2012,
        },
        "global_counts": {
            "total_images": total_images,
            "total_boxes": total_boxes,
            "total_points": total_points,
            "total_center_points": total_center_points,
            "total_corner_points": total_corner_points,
            "total_grid_cells_seen": total_grid_cells_seen,
        },
        "overflow_grid_counts": {
            "cells_total_points_gt_2B": grid_over_total,
            "cells_center_points_gt_B": grid_over_center,
            "cells_corner_points_gt_B": grid_over_corner,
        },
        "overflow_cell_composition": {
            "overflow_cells_total": overflow_cell_count,
            "single_class_overflow_cells": overflow_cell_single_class,
            "multi_class_overflow_cells": overflow_cell_multi_class,
            "single_class_ratio": single_ratio,
            "multi_class_ratio": multi_ratio,
        },
        "overflow_image_stats": {
            "overflow_images": overflow_image_count,
            "total_images": total_images,
            "overflow_image_rate": overflow_image_rate,
        },
        "class_frequency_all_points_sorted": class_counter_all_points.most_common(),
        "class_frequency_overflow_cells_sorted": class_counter_overflow_cells.most_common(),
        "class_ratio_in_multiclass_overflow_cells_sorted": [
            {
                "class": cls,
                "ratio": ratio,
                "count_in_multiclass_overflow_cells": num,
                "count_in_all_cells": den,
            }
            for cls, ratio, num, den in multi_overflow_class_ratio
        ],
        "overflow_image_examples": img_overflow_examples,
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"\nSaved stats json: {out}")


if __name__ == "__main__":
    main()

