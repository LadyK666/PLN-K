from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torchvision.transforms import functional as TF
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.pln_model import PLNModel  # noqa: E402
from datasets.voc_dataset import VOC2007Dataset  # noqa: E402
from utils.pln_channel_decoder import decode_branch_channels_inference  # noqa: E402
from utils.pln_candidate_pairs import generate_candidate_pairs_from_links  # noqa: E402
from utils.pln_pair_confidence import attach_pair_scores_and_labels_max_n  # noqa: E402
from utils.inference_geometry import uv_offset_to_image_xy, box_from_corner_and_center  # noqa: E402
from torch.utils.data import ConcatDataset
from utils.nms import class_aware_nms


def _boxes_from_center_corner(center_img_xy: torch.Tensor, corner_img_xy: torch.Tensor) -> torch.Tensor:
    # box_from_corner_and_center works with tensor [...,2]
    x_min, y_min, x_max, y_max = box_from_corner_and_center(corner_img_xy, center_img_xy)
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser("Debug pre-NMS inference pipeline (no NMS yet)")
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--use_voc2012", action="store_true")
    p.add_argument("--voc2007_root", type=str, default=str(ROOT / "VOC2007" / "VOCdevkit" / "VOC2007"))
    p.add_argument("--voc2012_root", type=str, default=str(ROOT / "VOC2012" / "VOCdevkit" / "VOC2012"))
    p.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint .pth")
    p.add_argument("--pretrained_backbone", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--branch_score_topk", type=int, default=200)
    p.add_argument("--conf_thres", type=float, default=0.25, help="Filter pair score (max_n prob) before NMS")
    p.add_argument("--nms_iou_threshold", type=float, default=0.45)
    p.add_argument("--nms_max_dets", type=int, default=60)
    p.add_argument("--visualize_nms", action="store_true", help="Draw final NMS boxes on the resized image")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = 14
    stride = args.image_size // S

    # Datasets (only for sampling images)
    ds_2007 = VOC2007Dataset(
        voc_root_dir=args.voc2007_root,
        split=args.split,
        output_size=(args.image_size, args.image_size),
        augment=False,
        dataset_tag="2007",
    )
    if args.use_voc2012:
        ds_2012 = VOC2007Dataset(
            voc_root_dir=args.voc2012_root,
            split=args.split,
            output_size=(args.image_size, args.image_size),
            augment=False,
            dataset_tag="2012",
        )
        ds = ConcatDataset([ds_2007, ds_2012])
    else:
        ds = ds_2007

    # Simple sampling: take first batch_size items
    images = []
    image_ids = []
    for idx in range(min(args.batch_size, len(ds))):
        images.append(ds[idx]["image"])
        image_ids.append(ds[idx]["target"]["image_id"])
    x = torch.stack(images, dim=0).to(device)  # (B,3,H,W)

    # Model
    model = PLNModel(
        backbone_pretrained=args.pretrained_backbone,
        backbone_trainable=False,
        freeze_bn=True,
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)

    model.eval()
    outs = model(x)  # dict of 4 branches

    branch_map = {
        "left_top": "left_top",
        "right_top": "right_top",
        "left_bottom": "left_bottom",
        "right_bottom": "right_bottom",
    }

    print(f"=== Pre-NMS inference debug ===")
    print(f"input: {tuple(x.shape)}, S={S}, stride={stride}")

    for branch_key, branch_name in branch_map.items():
        feat = outs[branch_key]  # (B,204,14,14)
        dec = decode_branch_channels_inference(feat, s=S)  # normalized P/Q/dx/dy/Lx/Ly

        pairs = generate_candidate_pairs_from_links(
            dec["P"],
            dec["Lx"],
            dec["Ly"],
            branch=branch_name,
            s=S,
            p_threshold=0.0,
        )
        if pairs["pairs_ijst"].shape[0] == 0:
            print(f"[{branch_key}] no candidate pairs")
            continue

        # score/label = max_n(prob)
        pairs = attach_pair_scores_and_labels_max_n(
            P=dec["P"],
            Q=dec["Q"],
            Lx=dec["Lx"],
            Ly=dec["Ly"],
            pairs_dict=pairs,
        )
        score = pairs["score"]  # (Np,)
        labels = pairs["label"]  # (Np,) in [0..N-1]

        if args.conf_thres > 0:
            keep = score >= args.conf_thres
            if keep.sum().item() == 0:
                print(f"[{branch_key}] no candidates after conf_thres={args.conf_thres}")
                continue
            score = score[keep]
            labels = labels[keep]
            for k in ["pairs_ijst", "src_xy", "tgt_xy", "src_point_type", "tgt_point_type",
                      "center_xy", "corner_xy", "center_point_type", "corner_point_type", "batch_idx"]:
                if k in pairs:
                    pairs[k] = pairs[k][keep]

        # Extract dx/dy for center/corner at their (u,v) grid cells
        batch_idx = pairs["batch_idx"]
        center_p = pairs["center_point_type"]
        corner_p = pairs["corner_point_type"]

        center_u = pairs["center_xy"][:, 0].round().to(torch.long)
        center_v = pairs["center_xy"][:, 1].round().to(torch.long)
        corner_u = pairs["corner_xy"][:, 0].round().to(torch.long)
        corner_v = pairs["corner_xy"][:, 1].round().to(torch.long)

        center_dx = dec["dx"][batch_idx, center_p, center_v, center_u]
        center_dy = dec["dy"][batch_idx, center_p, center_v, center_u]
        corner_dx = dec["dx"][batch_idx, corner_p, corner_v, corner_u]
        corner_dy = dec["dy"][batch_idx, corner_p, corner_v, corner_u]

        center_x_img, center_y_img = uv_offset_to_image_xy(
            u=center_u, v=center_v, dx=center_dx, dy=center_dy, stride=stride
        )
        corner_x_img, corner_y_img = uv_offset_to_image_xy(
            u=corner_u, v=corner_v, dx=corner_dx, dy=corner_dy, stride=stride
        )

        center_img_xy = torch.stack([center_x_img, center_y_img], dim=1)
        corner_img_xy = torch.stack([corner_x_img, corner_y_img], dim=1)

        boxes = _boxes_from_center_corner(center_img_xy=center_img_xy, corner_img_xy=corner_img_xy)
        # boxes: (Np,4)

        # Optional: keep top-k by score for readability
        if args.branch_score_topk > 0 and score.numel() > args.branch_score_topk:
            topk = torch.topk(score, k=args.branch_score_topk)
            sel = topk.indices
            boxes = boxes[sel]
            score = score[sel]
            labels = labels[sel]

        print(f"[{branch_key}] pairs={int(pairs['pairs_ijst'].shape[0])}, boxes_out={tuple(boxes.shape)}")
        print(f"[{branch_key}] score min/max: {float(score.min()):.6f} / {float(score.max()):.6f}")
        print(f"[{branch_key}] sample box[0]: {boxes[0].tolist() if boxes.numel() else None}")

        if boxes.numel() > 0:
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x_min_ok = float(x1.min())
            x_max_ok = float(x2.max())
            y_min_ok = float(y1.min())
            y_max_ok = float(y2.max())
            # Sanity checks
            has_nan = torch.isnan(boxes).any().item()
            has_inf = torch.isinf(boxes).any().item()
            inverted = ((x2 <= x1) | (y2 <= y1)).any().item()
            print(
                f"[{branch_key}] box ranges: x[{x_min_ok:.3f},{x_max_ok:.3f}] y[{y_min_ok:.3f},{y_max_ok:.3f}] "
                f"nan={has_nan} inf={has_inf} inverted={inverted}"
            )

        if "boxes_all" not in locals():
            boxes_all = []
            scores_all = []
            labels_all = []
        if boxes.numel() > 0:
            boxes_all.append(boxes.detach().cpu())
            scores_all.append(score.detach().cpu())
            labels_all.append(labels.detach().cpu())

    # ----- Run NMS on merged boxes from all 4 branches -----
    if "boxes_all" not in locals() or len(boxes_all) == 0:
        print("No boxes from any branch. NMS skipped.")
        return

    boxes_cat = torch.cat(boxes_all, dim=0)
    scores_cat = torch.cat(scores_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)

    print(f"=== Merged candidates: {boxes_cat.shape[0]} ===")

    keep_idx = class_aware_nms(
        boxes_cat, scores_cat, labels_cat,
        iou_threshold=args.nms_iou_threshold,
        max_dets=args.nms_max_dets,
    )

    final_boxes = boxes_cat[keep_idx]
    final_scores = scores_cat[keep_idx]
    final_labels = labels_cat[keep_idx]

    print(f"=== Final NMS output: {final_boxes.shape[0]} boxes ===")
    if final_boxes.shape[0] > 0:
        print("Final sample[0]:", final_boxes[0].tolist(), "label=", int(final_labels[0]), "score=", float(final_scores[0]))

    # -------- Visualization --------
    if args.visualize_nms:
        if args.batch_size != 1:
            raise ValueError("Visualization only supports --batch_size=1 right now.")
        if len(image_ids) != 1:
            raise ValueError("Internal error: image_ids not collected correctly.")

        # Prepare class names (VOC 20 classes)
        class_names = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor",
        ]

        img_id_full = image_ids[0]
        dataset_tag, raw_img_id = img_id_full.split("_", 1)
        if dataset_tag == "2007":
            ann_dataset = VOC2007Dataset(
                voc_root_dir=args.voc2007_root,
                split=args.split,
                output_size=(args.image_size, args.image_size),
                augment=False,
                dataset_tag="2007",
            )
        elif dataset_tag == "2012":
            ann_dataset = VOC2007Dataset(
                voc_root_dir=args.voc2012_root,
                split=args.split,
                output_size=(args.image_size, args.image_size),
                augment=False,
                dataset_tag="2012",
            )
        else:
            raise ValueError(f"Unknown dataset_tag in image_id: {dataset_tag} (from {img_id_full})")

        # Load original PIL and resize to match box coordinate system.
        pil = ann_dataset._load_image(raw_img_id)  # type: ignore[attr-defined]
        pil = TF.resize(pil, [args.image_size, args.image_size], antialias=True)

        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.imshow(pil)
        ax.axis("off")
        ax.set_title("NMS results")

        # Draw boxes
        for b, lab, sc in zip(final_boxes.cpu(), final_labels.cpu(), final_scores.cpu()):
            x1, y1, x2, y2 = b.tolist()
            w = x2 - x1
            h = y2 - y1
            cls = class_names[int(lab)] if 0 <= int(lab) < len(class_names) else str(int(lab))
            text = f"{cls}:{float(sc):.4f}"
            ax.add_patch(Rectangle((x1, y1), w, h, fill=False, edgecolor="red", linewidth=2))
            ax.text(x1, max(0.0, y1 - 2), text, color="yellow", fontsize=8, backgroundcolor="black")

        out_path = Path(args.voc2007_root).parent / "debug_out" / f"viz_nms_{img_id_full}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"NMS visualization saved to: {out_path}")


if __name__ == "__main__":
    main()

