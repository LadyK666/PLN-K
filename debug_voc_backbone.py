from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.voc_dataset import VOC2007Dataset  # noqa: E402
from models.backbone import BackBone  # noqa: E402
from models.pln_model import PLNModel  # noqa: E402


VOC2007_DEFAULT = str(Path(__file__).resolve().parent / "VOC2007" / "VOCdevkit" / "VOC2007")
VOC2012_DEFAULT = str(Path(__file__).resolve().parent / "VOC2012" / "VOCdevkit" / "VOC2012")


def _draw_boxes(ax, boxes_xyxy, labels, class_names: List[str]):
    for b, lab in zip(boxes_xyxy.tolist(), labels.tolist()):
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        cls = class_names[lab] if 0 <= lab < len(class_names) else str(lab)
        ax.add_patch(
            Rectangle((x1, y1), w, h, fill=False, edgecolor="red", linewidth=2)
        )
        ax.text(x1, max(0, y1 - 2), cls, color="yellow", fontsize=8, backgroundcolor="black")


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser("Debug VOC dataloader + BackBone shape")
    p.add_argument("--split", type=str, default="trainval", choices=["trainval", "train", "val", "test"])
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--tag", type=str, default="train")
    p.add_argument("--use_voc2012", action="store_true", help="Also load VOC2012 trainval for training stats")
    p.add_argument("--num_samples", type=int, default=20, help="<=0 means use all samples from selected splits")
    p.add_argument("--mix_datasets", action="store_true", help="Truly mix 2007 and 2012 in one shuffled list")
    p.add_argument("--viz", action="store_true", help="Save augmentation visualization")
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--pretrained_backbone", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parent / "debug_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset class names (must match loader)
    class_names = [
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

    ds_2007 = VOC2007Dataset(
        voc_root_dir=VOC2007_DEFAULT,
        split=args.split,
        output_size=(args.image_size, args.image_size),
        augment=True,
        dataset_tag="2007",
    )

    datasets = [ds_2007]
    if args.use_voc2012:
        ds_2012 = VOC2007Dataset(
            voc_root_dir=VOC2012_DEFAULT,
            split=args.split,
            output_size=(args.image_size, args.image_size),
            augment=True,
            dataset_tag="2012",
        )
        datasets.append(ds_2012)

    # Build an index list across datasets.
    # - if --mix_datasets: put (dataset_idx, sample_idx) into one list then shuffle.
    # - else: keep the original "dataset0 first then dataset1" order.
    all_items: List[Tuple[int, int]] = []
    if args.mix_datasets:
        for di, ds in enumerate(datasets):
            for i in range(len(ds)):
                all_items.append((di, i))
        # Shuffle to simulate "one total list" sampling.
        import random as _random
        _random.shuffle(all_items)
        if args.num_samples > 0:
            all_items = all_items[: args.num_samples]
    else:
        for di, ds in enumerate(datasets):
            cap = len(ds) if args.num_samples <= 0 else min(len(ds), args.num_samples)
            for i in range(max(0, cap)):
                all_items.append((di, i))

    box_counts: List[int] = []
    label_hist: Dict[int, int] = {}
    image_shapes: List[Tuple[int, ...]] = []

    for di, i in tqdm(all_items, desc="Sampling"):
        ds = datasets[di]
        sample = ds[i]
        img_t = sample["image"]  # (3,H,W)
        tgt = sample["target"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]

        image_shapes.append(tuple(img_t.shape))
        box_counts.append(int(boxes.shape[0]))

        for lab in labels.tolist():
            label_hist[lab] = label_hist.get(lab, 0) + 1

    # Print statistics
    print("=== Dataset shape stats ===")
    print("num_samples:", len(all_items))
    print("image_shapes:", image_shapes[:5], "... (unique=", sorted(set(image_shapes)), ")")
    print("box_counts: min/mean/max =", min(box_counts), sum(box_counts) / len(box_counts), max(box_counts))
    print("label_hist (top 10):", sorted(label_hist.items(), key=lambda x: x[1], reverse=True)[:10])
    # Additional debug: verify 2007/2012 mixed list composition.
    di_hist: Dict[int, int] = {}
    for di, _ in all_items:
        di_hist[di] = di_hist.get(di, 0) + 1
    print("dataset_index_hist (mixed order list):", di_hist)
    print("num_class_types_observed:", len(label_hist), " (VOC classes total =", 20, ")")

    # Visualization: show original vs augmented for first VOC2007 item
    if args.viz:
        ds = ds_2007
        img_id = ds.ids[0]
        image = ds._load_image(img_id)  # PIL
        boxes, labels = ds._load_target(img_id)
        assert isinstance(boxes, torch.Tensor)

        if ds.aug is not None:
            aug_image, aug_boxes, aug_labels = ds.aug(image, boxes, labels)
        else:
            aug_image = image
            aug_boxes, aug_labels = boxes, labels

        # Both images should be at ds.output_size after aug
        out_h, out_w = args.image_size, args.image_size
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].set_title("Original (before augmentation)")
        axes[1].set_title("After augmentation")

        axes[0].imshow(image)
        axes[1].imshow(aug_image)

        if boxes.numel() > 0:
            _draw_boxes(axes[0], boxes, labels, class_names=class_names)
        if aug_boxes.numel() > 0:
            _draw_boxes(axes[1], aug_boxes, aug_labels, class_names=class_names)

        for ax in axes:
            ax.axis("off")
        save_path = out_dir / f"viz_aug_2007_{img_id}.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved visualization to: {save_path}")

    # Backbone forward test: take a batch of one image
    # Note: BackBone expects (B,3,H,W)
    backbone = BackBone(pretrained=args.pretrained_backbone, requires_grad=False, freeze_bn=False).to(device)
    backbone.eval()

    sample = ds_2007[0]
    x = sample["image"].unsqueeze(0).to(device)
    out = backbone(x)
    print("=== BackBone forward test ===")
    print("input:", tuple(x.shape))
    print("output (layer4 feats):", tuple(out.shape))
    print("ok")

    # Full PLNModel forward test (adapter + 4 branches + dilation)
    model = PLNModel(
        backbone_pretrained=args.pretrained_backbone,
        backbone_trainable=False,
        freeze_bn=False,
    ).to(device)
    model.eval()
    outs = model(x)
    print("=== PLNModel forward test (adapter + 4 branches) ===")
    assert isinstance(outs, dict)
    for k in ["left_top", "right_top", "left_bottom", "right_bottom"]:
        v = outs[k]
        print(f"{k}: {tuple(v.shape)}")
    print("ok")


if __name__ == "__main__":
    main()

