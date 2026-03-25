from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.voc_dataset import VOC2007Dataset  # noqa: E402
from models.pln_model import PLNModel  # noqa: E402
from utils.pln_channel_decoder import decode_branch_channels_inference  # noqa: E402
from utils.pln_candidate_pairs import generate_candidate_pairs_from_links  # noqa: E402
from utils.pln_pair_confidence import attach_pair_scores_and_labels_max_n  # noqa: E402
from utils.inference_geometry import uv_offset_to_image_xy, box_from_corner_and_center  # noqa: E402
from utils.nms import class_aware_nms  # noqa: E402


VOC_CLASS_NAMES = [
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("PLN Inference (decoder -> candidates -> score/label -> NMS -> boxes)")
    p.add_argument(
        "--voc2007_root",
        type=str,
        default=str(ROOT / "VOC2007" / "VOCdevkit" / "VOC2007"),
        help="Path to VOCdevkit/VOC2007",
    )
    p.add_argument("--split", type=str, default="test", choices=["trainval", "train", "val", "test"])
    p.add_argument("--image_size", type=int, default=448, help="Final square size (H=W)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--checkpoint", type=str, default="", help="Optional .pth checkpoint from train.py")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_batches", type=int, default=0, help="0 means all batches")

    # Candidate filtering
    p.add_argument("--p_threshold", type=float, default=0.0, help="Filter candidates by existence P threshold")
    p.add_argument("--conf_thres", type=float, default=0.25, help="Filter pair score (max_n prob) before NMS")
    p.add_argument("--branch_score_topk", type=int, default=200, help="Top-k kept per branch before merging")

    # NMS
    p.add_argument("--nms_iou_threshold", type=float, default=0.2)
    p.add_argument("--nms_max_dets", type=int, default=10)

    # Post-NMS score filtering
    # Keep only boxes with score >= w * max_score_after_nms.
    p.add_argument(
        "--post_nms_score_ratio_filter",
        action="store_true",
        help="After NMS, further filter by score threshold: score >= w * s_max.",
    )
    p.add_argument(
        "--post_nms_score_ratio_w",
        type=float,
        default=0.35,
        help="The w in score >= w * s_max (used when --post_nms_score_ratio_filter is set).",
    )

    # Output
    p.add_argument("--output_dir", type=str, default=str(ROOT / "debug_out"))
    p.add_argument("--save_visualize", action="store_true", help="Save bbox visualization per image")
    p.add_argument("--log_pair_filter", action="store_true", help="Print per-branch candidate filtering logs")
    # Python 3.8 compat: BooleanOptionalAction is only available in newer Python versions.
    p.add_argument(
        "--eval_map",
        action="store_true",
        default=True,
        help="Compute mAP@IoU using VOC2007 11-point AP after inference (default: enabled).",
    )
    p.add_argument(
        "--no-eval_map",
        dest="eval_map",
        action="store_false",
        help="Disable mAP evaluation.",
    )
    p.add_argument("--map_iou_threshold", type=float, default=0.5, help="IoU threshold for AP/mAP (default 0.5)")
    return p.parse_args()


def _boxes_from_center_corner(center_img_xy: torch.Tensor, corner_img_xy: torch.Tensor) -> torch.Tensor:
    x_min, y_min, x_max, y_max = box_from_corner_and_center(corner_img_xy, center_img_xy)
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def _log_pair_stats(prefix: str, **kwargs) -> None:
    msg = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    print(f"[pair-log] {prefix} | {msg}")


def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    IoU for axis-aligned boxes in xyxy format.
    a: (N,4), b: (M,4) -> (N,M)
    """
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=torch.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = torch.maximum(ax1[:, None], bx1[None, :])
    iy1 = torch.maximum(ay1[:, None], by1[None, :])
    ix2 = torch.minimum(ax2[:, None], bx2[None, :])
    iy2 = torch.minimum(ay2[:, None], by2[None, :])
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    area_a = ((ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0))[:, None]
    area_b = ((bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0))[None, :]
    union = (area_a + area_b - inter).clamp(min=1e-6)
    return inter / union


def _voc07_ap(rec: torch.Tensor, prec: torch.Tensor) -> float:
    """
    VOC2007 11-point AP computation.
    """
    ap = 0.0
    for t in torch.arange(0.0, 1.1, 0.1, device=rec.device):
        p = prec[rec >= t].max() if (rec >= t).any() else torch.tensor(0.0, device=rec.device)
        ap += float(p.item()) / 11.0
    return ap


def _compute_map_voc07(
    *,
    predictions: List[Dict],
    gt_by_image_id: Dict[str, Dict[str, torch.Tensor]],
    num_classes: int,
    iou_thres: float,
) -> Dict:
    """
    Compute mAP and per-class AP using VOC2007 11-point AP at given IoU threshold.

    predictions: list of dicts from this script, each contains:
      - image_id
      - boxes_xyxy (list)
      - scores (list)
      - labels (list)
    gt_by_image_id: image_id -> {"boxes": (Ng,4), "labels": (Ng,)}
    """
    device = torch.device("cpu")

    # Pre-convert all predictions to a flat list per class (small scale: 50 images, <=10 det/image).
    pred_by_class: List[List[Dict[str, object]]] = [[] for _ in range(num_classes)]
    for pred in predictions:
        img_id = pred["image_id"]
        boxes_list = pred.get("boxes_xyxy", [])
        scores_list = pred.get("scores", [])
        labels_list = pred.get("labels", [])
        if not boxes_list:
            continue
        boxes_t = torch.tensor(boxes_list, dtype=torch.float32, device=device)
        scores_t = torch.tensor(scores_list, dtype=torch.float32, device=device)
        labels_t = torch.tensor(labels_list, dtype=torch.long, device=device)
        for box, score, lab in zip(boxes_t, scores_t, labels_t):
            li = int(lab.item())
            if 0 <= li < num_classes:
                pred_by_class[li].append(
                    {"image_id": img_id, "box": box, "score": float(score.item())}
                )

    # Pre-group GT boxes by class and image for faster matching.
    gt_boxes_by_class: List[Dict[str, torch.Tensor]] = [dict() for _ in range(num_classes)]
    gt_matched_by_class: List[Dict[str, torch.Tensor]] = [dict() for _ in range(num_classes)]
    gt_count_by_class: List[int] = [0 for _ in range(num_classes)]

    for img_id, gt in gt_by_image_id.items():
        gt_boxes = gt["boxes"].to(device=device, dtype=torch.float32)
        gt_labels = gt["labels"].to(device=device, dtype=torch.long)
        for c in range(num_classes):
            mask = gt_labels == c
            boxes_c = gt_boxes[mask]
            gt_boxes_by_class[c][img_id] = boxes_c
            gt_matched_by_class[c][img_id] = torch.zeros((boxes_c.shape[0],), dtype=torch.bool)
            gt_count_by_class[c] += int(boxes_c.shape[0])

    ap_per_class: List[float] = [0.0 for _ in range(num_classes)]
    for c in range(num_classes):
        preds_c = pred_by_class[c]
        preds_c.sort(key=lambda x: x["score"], reverse=True)

        n_gt = gt_count_by_class[c]
        if n_gt == 0:
            ap_per_class[c] = 0.0
            continue

        tp = torch.zeros((len(preds_c),), dtype=torch.float32, device=device)
        fp = torch.zeros((len(preds_c),), dtype=torch.float32, device=device)

        # Reset per-class matched flags.
        matched_flags = gt_matched_by_class[c]

        for i, p in enumerate(preds_c):
            img_id = str(p["image_id"])
            box_p: torch.Tensor = p["box"]  # (4,)
            if img_id not in gt_boxes_by_class[c]:
                fp[i] = 1.0
                continue
            gt_boxes_c = gt_boxes_by_class[c][img_id]
            if gt_boxes_c.numel() == 0:
                fp[i] = 1.0
                continue

            ious = _box_iou_xyxy(box_p.view(1, 4), gt_boxes_c).view(-1)
            best_iou, best_j = ious.max(dim=0)
            best_j_int = int(best_j.item())
            if float(best_iou.item()) >= iou_thres and not bool(matched_flags[img_id][best_j_int].item()):
                tp[i] = 1.0
                matched_flags[img_id][best_j_int] = torch.tensor(True)
            else:
                fp[i] = 1.0

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        rec = tp_cum / float(n_gt)
        prec = tp_cum / (tp_cum + fp_cum + 1e-6)
        ap_per_class[c] = _voc07_ap(rec=rec, prec=prec)

    map_ = float(sum(ap_per_class) / num_classes) if num_classes > 0 else 0.0

    return {
        "mAP": map_,
        "mAP_iou_thres": float(iou_thres),
        "ap_per_class": ap_per_class,
        "class_names": VOC_CLASS_NAMES[:num_classes],
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    S = 14
    stride = args.image_size // S

    ds = VOC2007Dataset(
        voc_root_dir=args.voc2007_root,
        split=args.split,
        output_size=(args.image_size, args.image_size),
        augment=False,
        dataset_tag="2007",
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: {
            "images": torch.stack([b["image"] for b in batch], dim=0),
            "targets": [b["target"] for b in batch],
        },
    )

    model = PLNModel(backbone_pretrained=False, backbone_trainable=False, freeze_bn=True).to(device)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)

    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions: List[Dict] = []
    gt_by_image_id: Dict[str, Dict[str, torch.Tensor]] = {}

    branch_map = {
        "left_top": "left_top",
        "right_top": "right_top",
        "left_bottom": "left_bottom",
        "right_bottom": "right_bottom",
    }

    # Progress bar over batches
    total_batches = int((len(ds) + args.batch_size - 1) // args.batch_size)  # includes last partial batch
    if args.max_batches > 0:
        total_batches = min(total_batches, args.max_batches)

    for bi, batch in tqdm(enumerate(dl), total=total_batches, desc="Inference", unit="batch"):
        if args.max_batches > 0 and bi >= args.max_batches:
            break

        images: torch.Tensor = batch["images"].to(device)  # (B,3,H,W)
        targets = batch["targets"]
        B = images.shape[0]

        # Collect GT for mAP evaluation (boxes/labels are already in resized pixel coords).
        for b in range(B):
            t = targets[b]
            img_id = t["image_id"]
            gt_by_image_id[str(img_id)] = {
                "boxes": t["boxes"],
                "labels": t["labels"],
            }

        outs = model(images)

        boxes_all: List[torch.Tensor] = []
        scores_all: List[torch.Tensor] = []
        labels_all: List[torch.Tensor] = []
        batch_idx_all: List[torch.Tensor] = []

        for branch_key, branch_name in branch_map.items():
            feat = outs[branch_key]  # (B,204,14,14)
            dec = decode_branch_channels_inference(feat, s=S)  # normalized

            branch_raw = int(dec["P"].shape[0] * dec["P"].shape[1] * dec["P"].shape[2] * dec["P"].shape[3])
            pairs = generate_candidate_pairs_from_links(
                dec["P"],
                dec["Lx"],
                dec["Ly"],
                branch=branch_name,
                s=S,
                p_threshold=args.p_threshold,
            )
            n_after_geometry = int(pairs["pairs_ijst"].shape[0])
            if n_after_geometry == 0:
                if args.log_pair_filter:
                    _log_pair_stats(
                        f"batch={bi} branch={branch_key}",
                        raw_points=branch_raw,
                        kept_after_geometry=0,
                        dropped_geometry=branch_raw,
                        reason="geometry/p_threshold produced no valid pairs",
                    )
                continue

            pairs = attach_pair_scores_and_labels_max_n(
                P=dec["P"],
                Q=dec["Q"],
                Lx=dec["Lx"],
                Ly=dec["Ly"],
                pairs_dict=pairs,
            )

            score = pairs["score"]  # (Np,)
            labels = pairs["label"]  # (Np,)
            n_before_conf = int(score.numel())

            # Score threshold (pair confidence)
            # Note: we do not count conf_thres as an elimination reason in logs here.
            if args.conf_thres > 0:
                keep = score >= args.conf_thres
                if keep.sum().item() == 0:
                    if args.log_pair_filter:
                        _log_pair_stats(
                            f"batch={bi} branch={branch_key}",
                            raw_points=branch_raw,
                            kept_after_geometry=n_after_geometry,
                            dropped_geometry=branch_raw - n_after_geometry,
                            kept_before_topk=0,
                            dropped_by_topk=0,
                            reason="all removed by conf_thres (excluded from elimination stats)",
                        )
                    continue
                score = score[keep]
                labels = labels[keep]
                for k in ["pairs_ijst", "src_xy", "tgt_xy", "src_point_type", "tgt_point_type",
                          "center_xy", "corner_xy", "center_point_type", "corner_point_type", "batch_idx"]:
                    if k in pairs:
                        pairs[k] = pairs[k][keep]

            # Top-k per branch
            n_before_topk = int(score.numel())
            if args.branch_score_topk > 0 and score.numel() > args.branch_score_topk:
                topk = torch.topk(score, k=args.branch_score_topk)
                sel = topk.indices
                score = score[sel]
                labels = labels[sel]
                for k in ["pairs_ijst", "src_xy", "tgt_xy", "src_point_type", "tgt_point_type",
                          "center_xy", "corner_xy", "center_point_type", "corner_point_type", "batch_idx"]:
                    if k in pairs:
                        pairs[k] = pairs[k][sel]
            n_after_topk = int(score.numel())

            # Candidate -> boxes
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

            # Clip to image
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0.0, max=float(args.image_size))
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0.0, max=float(args.image_size))

            boxes_all.append(boxes)
            scores_all.append(score)
            labels_all.append(labels)
            batch_idx_all.append(batch_idx)

            if args.log_pair_filter:
                _log_pair_stats(
                    f"batch={bi} branch={branch_key}",
                    raw_points=branch_raw,
                    kept_after_geometry=n_after_geometry,
                    dropped_geometry=branch_raw - n_after_geometry,
                    kept_before_topk=n_before_topk,
                    dropped_by_topk=max(0, n_before_topk - n_after_topk),
                    kept_after_topk=n_after_topk,
                    note=f"conf_kept={n_before_topk}/{n_before_conf} (conf not counted as elimination reason)",
                )

        if len(boxes_all) == 0:
            continue

        boxes_cat = torch.cat(boxes_all, dim=0)
        scores_cat = torch.cat(scores_all, dim=0)
        labels_cat = torch.cat(labels_all, dim=0)
        batch_idx_cat = torch.cat(batch_idx_all, dim=0)

        for b in range(B):
            mask = batch_idx_cat == b
            if mask.sum().item() == 0:
                continue

            boxes_b = boxes_cat[mask]
            scores_b = scores_cat[mask]
            labels_b = labels_cat[mask]
            n_before_nms = int(boxes_b.shape[0])

            keep_idx = class_aware_nms(
                boxes_b,
                scores_b,
                labels_b,
                iou_threshold=args.nms_iou_threshold,
                max_dets=args.nms_max_dets,
            )

            kept_boxes = boxes_b[keep_idx]
            kept_scores = scores_b[keep_idx]
            kept_labels = labels_b[keep_idx]
            n_after_nms = int(kept_boxes.shape[0])
            n_after_nms_before_score = n_after_nms

            # Optional post-NMS filtering by relative score.
            # This is independent of class-aware NMS; it simply trims the tail.
            if args.post_nms_score_ratio_filter and kept_scores.numel() > 0:
                s_max = float(kept_scores.max().item())
                w = float(args.post_nms_score_ratio_w)
                score_thres = w * s_max
                keep_score = kept_scores >= score_thres
                kept_boxes = kept_boxes[keep_score]
                kept_scores = kept_scores[keep_score]
                kept_labels = kept_labels[keep_score]
                # (Optional) overwrite stats for logs/predictions.
                n_after_nms = int(kept_boxes.shape[0])
            n_after_score = n_after_nms

            if args.log_pair_filter:
                _log_pair_stats(
                    f"batch={bi} image_idx={b} nms",
                    before_nms=n_before_nms,
                    after_nms=n_after_nms_before_score,
                    after_score_filter=n_after_score,
                    dropped_by_nms=max(0, n_before_nms - n_after_nms_before_score),
                    dropped_by_score_filter=max(0, n_after_nms_before_score - n_after_score),
                )

            img_id = targets[b]["image_id"]
            predictions.append(
                {
                    "image_id": img_id,
                    "boxes_xyxy": kept_boxes.cpu().tolist(),
                    "scores": kept_scores.cpu().tolist(),
                    "labels": kept_labels.cpu().tolist(),
                }
            )

            if args.save_visualize:
                dataset_tag, raw_img_id = img_id.split("_", 1)
                pil = ds._load_image(raw_img_id)  # type: ignore[attr-defined]
                pil = TF.resize(pil, [args.image_size, args.image_size], antialias=True)

                import matplotlib.pyplot as plt
                from matplotlib.patches import Rectangle

                fig, ax = plt.subplots(1, 1, figsize=(9, 9))
                ax.imshow(pil)
                ax.axis("off")
                ax.set_title("PLN NMS")

                for bb, ll, sc in zip(kept_boxes.cpu(), kept_labels.cpu(), kept_scores.cpu()):
                    x1, y1, x2, y2 = bb.tolist()
                    w = x2 - x1
                    h = y2 - y1
                    cls = VOC_CLASS_NAMES[int(ll)] if 0 <= int(ll) < len(VOC_CLASS_NAMES) else str(int(ll))
                    ax.add_patch(Rectangle((x1, y1), w, h, fill=False, edgecolor="red", linewidth=2))
                    ax.text(
                        x1,
                        max(0.0, y1 - 2),
                        f"{cls}:{float(sc):.4f}",
                        color="yellow",
                        fontsize=8,
                        backgroundcolor="black",
                    )

                out_path = out_dir / f"viz_nms_{img_id}.png"
                fig.tight_layout()
                fig.savefig(out_path, dpi=150)
                plt.close(fig)

                # Additional: score distribution bar plot after NMS.
                # By default NMS keeps at most `args.nms_max_dets` boxes per image,
                # so we try to visualize exactly that many bars.
                kept_scores_cpu = kept_scores.detach().cpu()
                K = int(args.nms_max_dets)
                if K > 0:
                    scores_sorted, _ = torch.sort(kept_scores_cpu, descending=True)
                    n = int(scores_sorted.numel())
                    if n < K:
                        # Pad with zeros so the bar count stays consistent (always K bars).
                        pad = torch.zeros((K - n,), dtype=scores_sorted.dtype)
                        scores_for_plot = torch.cat([scores_sorted, pad], dim=0)
                        padded_mask = [False] * n + [True] * (K - n)
                    else:
                        scores_for_plot = scores_sorted[:K]
                        padded_mask = [False] * K

                    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 3.5))
                    bar_colors = ["tab:blue" if not is_pad else "lightgray" for is_pad in padded_mask]
                    ax2.bar(list(range(K)), scores_for_plot.tolist(), color=bar_colors, width=0.8)
                    ax2.set_title(f"NMS score distribution (top-{K})")
                    ax2.set_xlabel("Kept box rank (after score sort)")
                    ax2.set_ylabel("score")
                    ax2.grid(axis="y", linestyle="--", alpha=0.4)

                    out_path2 = out_dir / f"score_bar_nms_{img_id}.png"
                    fig2.tight_layout()
                    fig2.savefig(out_path2, dpi=150)
                    plt.close(fig2)

    # Ensure output dir exists right before writing (robust against mid-run issues).
    if out_dir.exists() and out_dir.is_file():
        raise NotADirectoryError(f"Output path exists but is a file: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"predictions_{args.split}.json"
    with open(out_json, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions to: {out_json} (num_images={len(predictions)})")

    if args.eval_map:
        # COCO-style multi-threshold mAP:
        # IoU thresholds = [0.50, 0.55, ..., 0.95] (10 values)
        iou_thresholds = [round(0.5 + 0.05 * k, 2) for k in range(10)]
        num_classes = len(VOC_CLASS_NAMES)

        per_iou_results: Dict[str, Dict] = {}
        for t in iou_thresholds:
            map_res = _compute_map_voc07(
                predictions=predictions,
                gt_by_image_id=gt_by_image_id,
                num_classes=num_classes,
                iou_thres=float(t),
            )
            per_iou_results[str(t)] = map_res

        class_names = per_iou_results[str(iou_thresholds[0])]["class_names"]

        # mAP@0.5 and mAP@0.75 (mean over classes).
        res_05 = per_iou_results["0.5"]
        res_075 = per_iou_results["0.75"]
        ap_05 = res_05["ap_per_class"]
        ap_075 = res_075["ap_per_class"]

        mAP_05 = float(res_05["mAP"])
        mAP_075 = float(res_075["mAP"])

        # mAP@[0.5:0.95]:
        #  - compute AP per class for each IoU threshold
        #  - then average per class across IoU thresholds
        #  - finally mAP is the mean across classes.
        ap_matrix = torch.tensor([per_iou_results[str(t)]["ap_per_class"] for t in iou_thresholds], dtype=torch.float32)
        ap_per_class_avg = ap_matrix.mean(dim=0).tolist()  # (C,)
        mAP_50595 = float(torch.tensor(ap_per_class_avg, dtype=torch.float32).mean().item())

        print("\n[Eval] mAP@0.50 = %.6f" % mAP_05)
        for c, ap in enumerate(ap_05):
            print("  AP@0.50 %-12s = %.6f" % (class_names[c], ap))

        print("\n[Eval] mAP@0.75 = %.6f" % mAP_075)
        for c, ap in enumerate(ap_075):
            print("  AP@0.75 %-12s = %.6f" % (class_names[c], ap))

        print("\n[Eval] mAP@[0.50:0.95] = %.6f (10 IoU thresholds)" % mAP_50595)
        for c, ap in enumerate(ap_per_class_avg):
            print("  AP@avg IoU %-12s = %.6f" % (class_names[c], ap))

        # Save everything in one json.
        out_map_json = out_dir / f"map_results_{args.split}.json"
        out_obj = {
            "metrics": {
                "mAP@0.5": mAP_05,
                "mAP@0.75": mAP_075,
                "mAP@[0.5:0.95]": mAP_50595,
                "iou_thresholds": iou_thresholds,
                "ap_eval_style": "VOC2007_11point_ap",
            },
            "ap_per_class": {
                "at_iou_thresholds": {
                    str(t): {
                        "mAP": float(per_iou_results[str(t)]["mAP"]),
                        "ap_per_class": [float(x) for x in per_iou_results[str(t)]["ap_per_class"]],
                    }
                    for t in iou_thresholds
                },
                "at_0.5": [float(x) for x in ap_05],
                "at_0.75": [float(x) for x in ap_075],
                "avg_over_iou_thresholds_[0.5:0.95]": [float(x) for x in ap_per_class_avg],
            },
            "class_names": class_names,
        }
        with open(out_map_json, "w") as f:
            json.dump(out_obj, f, indent=2)

        print(f"[Eval] Saved mAP results to: {out_map_json}")


if __name__ == "__main__":
    main()

