from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


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

CLASS_PRIORITY = [
    "diningtable",
    "bicycle",
    "bottle",
    "chair",
    "motorbike",
    "car",
    "person",
    "bus",
    "horse",
    "pottedplant",
    "tvmonitor",
    "boat",
    "cow",
    "sofa",
    "bird",
    "aeroplane",
    "train",
    "sheep",
    "dog",
    "cat",
]

_PRIORITY_RANK = {c: i for i, c in enumerate(CLASS_PRIORITY)}
_CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _to_grid_and_incell(
    x: float,
    y: float,
    *,
    src_w: float,
    src_h: float,
    image_size: int,
    grid_size: int,
):
    x_r = x / src_w * image_size
    y_r = y / src_h * image_size
    stride = image_size / grid_size
    u = _clamp_int(int(x_r // stride), 0, grid_size - 1)
    v = _clamp_int(int(y_r // stride), 0, grid_size - 1)
    # Right-top corner referenced offsets:
    # x_rt=(u+1)*stride, y_rt=v*stride
    # dx=(x_rt-x)/stride, dy=(y-y_rt)/stride
    x_rt = (u + 1) * stride
    y_rt = v * stride
    dx = (x_rt - x_r) / stride
    dy = (y_r - y_rt) / stride
    dx = float(max(0.0, min(1.0, dx)))
    dy = float(max(0.0, min(1.0, dy)))
    return x_r, y_r, u, v, dx, dy


def _pick_branch_corner_xy(x1: float, y1: float, x2: float, y2: float, branch: str):
    if branch == "left_top":
        return x1, y1
    if branch == "right_top":
        return x2, y1
    if branch == "left_bottom":
        return x1, y2
    if branch == "right_bottom":
        return x2, y2
    raise ValueError(f"Unknown branch={branch}")


def _point_sort_key(cls: str, area: float):
    return (_PRIORITY_RANK.get(cls, 10**9), float(area))


def _select_top2(points):
    if len(points) <= 2:
        return points
    points_sorted = sorted(points, key=lambda p: _point_sort_key(p["cls"], p["area"]))
    return points_sorted[:2]


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


def _pick_image_id(voc_root: Path, split: str, image_id: str) -> str:
    if image_id:
        return image_id
    ids = _read_split_ids(voc_root, split)
    if not ids:
        raise ValueError(f"No ids found in split={split} under {voc_root}")
    return random.choice(ids)


def main() -> None:
    p = argparse.ArgumentParser("Visualize boxes + 14x14 grid overlay")
    p.add_argument(
        "--voc_root",
        type=str,
        default="",
        help="Path to VOCdevkit/VOC2007 (e.g. /path/to/VOCdevkit/VOC2007). Required.",
    )
    p.add_argument("--split", type=str, default="trainval")
    p.add_argument("--image_id", type=str, default="", help="If empty, random sample from split")
    p.add_argument("--image_size", type=int, default=448, help="Resize image to image_size x image_size")
    p.add_argument("--grid_size", type=int, default=14, help="Grid divisions for H/W")
    p.add_argument(
        "--branch",
        type=str,
        default="left_top",
        choices=["left_top", "right_top", "left_bottom", "right_bottom"],
        help="Branch for corner filtering / link target construction",
    )
    p.add_argument("--output", type=str, default="debug_out/grid_overlay_sample.png")
    args = p.parse_args()

    if not args.voc_root:
        raise ValueError("--voc_root is required (no machine-specific default is bundled).")
    voc_root = Path(args.voc_root)
    image_id = _pick_image_id(voc_root, args.split, args.image_id)

    img_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    boxes, labels = _parse_boxes_and_labels(voc_root, image_id)

    image = Image.open(img_path).convert("RGB")
    src_w, src_h = image.size
    image_resized = image.resize((args.image_size, args.image_size))
    scale_x = args.image_size / float(src_w)
    scale_y = args.image_size / float(src_h)

    # Build targets in pure python (no torch) and prepare a compact text summary.
    # Slot convention:
    #   - p0,p1: centers (j=1,2)
    #   - p2,p3: corners (j=3,4), but only the branch-specific corner is considered per box
    S = args.grid_size
    stride = args.image_size / float(S)

    centers_by_cell = {}
    corners_by_cell = {}

    for (x1, y1, x2, y2), cls in zip(boxes, labels):
        if cls not in _CLASS_TO_IDX:
            continue
        area = float(max(0.0, (x2 - x1) * (y2 - y1)))

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        _, _, u_c, v_c, dx_c, dy_c = _to_grid_and_incell(
            cx, cy, src_w=src_w, src_h=src_h, image_size=args.image_size, grid_size=S
        )

        kx, ky = _pick_branch_corner_xy(x1, y1, x2, y2, args.branch)
        _, _, u_k, v_k, dx_k, dy_k = _to_grid_and_incell(
            kx, ky, src_w=src_w, src_h=src_h, image_size=args.image_size, grid_size=S
        )

        centers_by_cell.setdefault((u_c, v_c), []).append(
            {"cls": cls, "area": area, "dx": dx_c, "dy": dy_c, "link_u": u_k, "link_v": v_k}
        )
        corners_by_cell.setdefault((u_k, v_k), []).append(
            {"cls": cls, "area": area, "dx": dx_k, "dy": dy_k, "link_u": u_c, "link_v": v_c}
        )

    selected = []  # list of dict {p,u,v,cls,dx,dy,link_u,link_v}

    # centers
    for (u, v), pts in centers_by_cell.items():
        chosen = _select_top2(pts)
        for slot, pnt in enumerate(chosen):
            selected.append(
                {"p": slot, "u": u, "v": v, **pnt}
            )

    # corners
    for (u, v), pts in corners_by_cell.items():
        chosen = _select_top2(pts)
        for slot, pnt in enumerate(chosen):
            selected.append(
                {"p": 2 + slot, "u": u, "v": v, **pnt}
            )

    lines = []
    lines.append(f"branch={args.branch}  |  slots: p0-1=center, p2-3=corner({args.branch})")
    lines.append("Format per active cell: (u,v) p#: class  (x,y)  link->(u_t,v_t)")

    # limit text so the figure stays readable
    max_lines = 24
    # sort for stable display: by (v,u,p)
    selected = sorted(selected, key=lambda d: (d["v"], d["u"], d["p"]))
    for idx in range(min(len(selected), max_lines)):
        d = selected[idx]
        lines.append(
            f"({d['u']:02d},{d['v']:02d}) p{d['p']}: {d['cls']:11s} ({d['dx']:.2f},{d['dy']:.2f}) link->({d['link_u']:02d},{d['link_v']:02d})"
        )
    if len(selected) > max_lines:
        lines.append(f"... truncated: {len(selected) - max_lines} more active slots ...")

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.6, 1.4])
    ax = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[1, 0])

    ax.imshow(image_resized)

    # Draw bboxes after resize
    for (x1, y1, x2, y2), cls in zip(boxes, labels):
        rx1, ry1 = x1 * scale_x, y1 * scale_y
        rx2, ry2 = x2 * scale_x, y2 * scale_y
        w, h = rx2 - rx1, ry2 - ry1
        rect = Rectangle((rx1, ry1), w, h, fill=False, linewidth=1.8, edgecolor="lime")
        ax.add_patch(rect)
        ax.text(
            rx1,
            max(0.0, ry1 - 3),
            cls,
            fontsize=7,
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.45, pad=1.2, edgecolor="none"),
        )

    # Draw 14x14 grid (equal divisions)
    stride = args.image_size / float(args.grid_size)
    for i in range(args.grid_size + 1):
        x = i * stride
        y = i * stride
        ax.plot([x, x], [0, args.image_size], color="cyan", linewidth=0.8, alpha=0.8)
        ax.plot([0, args.image_size], [y, y], color="cyan", linewidth=0.8, alpha=0.8)

    ax.set_title(f"{image_id} | boxes + {args.grid_size}x{args.grid_size} grid | size={args.image_size}")
    ax.set_xlim([0, args.image_size])
    ax.set_ylim([args.image_size, 0])
    ax.axis("off")

    ax_text.axis("off")
    ax_text.text(
        0.01,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

    print(f"Saved visualization: {out}")
    print(f"Image id: {image_id}")
    print(f"Num boxes: {len(boxes)}")


if __name__ == "__main__":
    main()

