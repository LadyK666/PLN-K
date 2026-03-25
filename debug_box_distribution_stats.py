from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt


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


def _parse_boxes_and_labels(voc_root: Path, img_id: str) -> Tuple[List[Tuple[float, float, float, float]], List[str]]:
    xml_path = voc_root / "Annotations" / f"{img_id}.xml"
    root = ET.parse(xml_path).getroot()

    boxes: List[Tuple[float, float, float, float]] = []
    labels: List[str] = []

    for obj in root.findall("object"):
        name_node = obj.find("name")
        bb = obj.find("bndbox")
        if name_node is None or bb is None:
            continue
        cls = (name_node.text or "").strip()
        if cls not in VOC_CLASSES:
            continue

        x1 = float(bb.find("xmin").text)  # type: ignore[union-attr]
        y1 = float(bb.find("ymin").text)  # type: ignore[union-attr]
        x2 = float(bb.find("xmax").text)  # type: ignore[union-attr]
        y2 = float(bb.find("ymax").text)  # type: ignore[union-attr]

        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
        labels.append(cls)

    return boxes, labels


def _freq_bar(counts: List[int]) -> Dict[int, float]:
    # return {k: freq_percentage}
    n = len(counts)
    c = Counter(counts)
    return {k: (v / n) * 100.0 for k, v in sorted(c.items())}


def _freq_bar_nonzero(counts: List[int]) -> Dict[int, float]:
    # Only consider images where this class appears (count > 0)
    nz = [x for x in counts if x > 0]
    if not nz:
        return {}
    n = len(nz)
    c = Counter(nz)
    return {k: (v / n) * 100.0 for k, v in sorted(c.items())}


def _plot_overall_and_per_class(
    out_png: Path,
    total_counts_per_image: List[int],
    per_class_counts_per_image: Dict[str, List[int]],
) -> None:
    n_classes = len(VOC_CLASSES)
    cols = 5
    rows = (n_classes + cols - 1) // cols  # should be 4 for 20 classes

    fig = plt.figure(figsize=(18, 6 + rows * 2.2))
    gs = fig.add_gridspec(rows + 1, cols)

    # Main plot: overall #boxes per image
    ax_main = fig.add_subplot(gs[0, :])
    freq_total = _freq_bar(total_counts_per_image)
    xs = list(freq_total.keys())
    ys = [freq_total[x] for x in xs]
    ax_main.bar(xs, ys, width=0.9, color="#4c78a8")
    ax_main.set_title("#Boxes per Image (All Classes) - % frequency")
    ax_main.set_xlabel("Number of boxes per image")
    ax_main.set_ylabel("Frequency (%)")
    ax_main.grid(axis="y", alpha=0.25)

    # Subplots: per class distribution of #boxes per image
    for idx, cls in enumerate(VOC_CLASSES):
        r = 1 + idx // cols
        c = idx % cols
        ax = fig.add_subplot(gs[r, c])

        freq = _freq_bar_nonzero(per_class_counts_per_image[cls])
        if freq:
            xs_cls = list(freq.keys())
            ys_cls = [freq[x] for x in xs_cls]
            ax.bar(xs_cls, ys_cls, width=0.9, color="#54a24b")
        else:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", fontsize=8, transform=ax.transAxes)

        ax.set_title(cls, fontsize=9)
        ax.set_xlabel("# boxes/image", fontsize=8)
        ax.set_ylabel("%", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser("Box-count distribution stats for VOC trainval")
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
    p.add_argument("--use_voc2012", action="store_true", help="Include VOC2012 trainval")
    p.add_argument("--split", type=str, default="trainval")
    p.add_argument("--output_dir", type=str, default="debug_out/box_distribution")
    args = p.parse_args()

    if not args.voc2007_root:
        raise ValueError("--voc2007_root is required (no machine-specific default is bundled).")
    voc_roots = [Path(args.voc2007_root)]
    if args.use_voc2012:
        if not args.voc2012_root:
            raise ValueError("--voc2012_root is required when --use_voc2012 is set.")
        voc_roots.append(Path(args.voc2012_root))

    total_images = 0
    total_boxes = 0

    total_counts_per_image: List[int] = []
    per_class_counts_per_image: Dict[str, List[int]] = {cls: [] for cls in VOC_CLASSES}

    # For global sanity / reporting
    per_class_total_boxes = Counter()

    for root in voc_roots:
        ids = _read_split_ids(root, args.split)
        for img_id in ids:
            boxes, labels = _parse_boxes_and_labels(root, img_id)

            total_images += 1
            total_boxes += len(boxes)
            total_counts_per_image.append(len(boxes))

            # initialize per-class counts for this image with 0
            per_img_cls_count = Counter(labels)
            for cls in VOC_CLASSES:
                per_class_counts_per_image[cls].append(per_img_cls_count.get(cls, 0))
            per_class_total_boxes.update(per_img_cls_count)

    overall_mean_boxes_per_image = (total_boxes / total_images) if total_images > 0 else 0.0

    per_class_mean_boxes_per_image = {}
    for cls in VOC_CLASSES:
        arr = per_class_counts_per_image[cls]
        per_class_mean_boxes_per_image[cls] = (sum(arr) / total_images) if total_images > 0 else 0.0

    out_dir = Path(args.output_dir)
    out_png = out_dir / "box_counts_main_and_class_subplots.png"
    out_json = out_dir / "box_counts_stats.json"

    _plot_overall_and_per_class(
        out_png=out_png,
        total_counts_per_image=total_counts_per_image,
        per_class_counts_per_image=per_class_counts_per_image,
    )

    # Save JSON for exact values
    result = {
        "config": {
            "voc2007_root": str(Path(args.voc2007_root)),
            "voc2012_root": str(Path(args.voc2012_root)),
            "use_voc2012": args.use_voc2012,
            "split": args.split,
            "num_classes": len(VOC_CLASSES),
        },
        "global_counts": {
            "total_images": total_images,
            "total_boxes": total_boxes,
            "mean_boxes_per_image": overall_mean_boxes_per_image,
        },
        "per_class_counts": {
            "per_class_total_boxes": dict(per_class_total_boxes),
            "per_class_mean_boxes_per_image": per_class_mean_boxes_per_image,
        },
        "output": {
            "png": str(out_png),
            "json": str(out_json),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))

    # Also print key values
    print(json.dumps(result["global_counts"], indent=2))
    print("\nTop classes by per_class_mean_boxes_per_image (top 8):")
    top = sorted(per_class_mean_boxes_per_image.items(), key=lambda x: x[1], reverse=True)[:8]
    for cls, m in top:
        print(f"- {cls}: {m:.4f}")
    print(f"\nSaved:\n- {out_png}\n- {out_json}")


if __name__ == "__main__":
    main()

