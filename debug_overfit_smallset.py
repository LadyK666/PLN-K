from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.collate_detection import collate_voc_detection  # noqa: E402
from datasets.voc_dataset import VOC2007Dataset  # noqa: E402
from models.pln_model import PLNModel  # noqa: E402
from utils.pln_channel_decoder import decode_branch_channels_logits, decode_branch_channels_inference  # noqa: E402
from utils.pln_loss import PLNLoss, PLNLossWeights  # noqa: E402
from utils.pln_target_builder import build_pln_targets_for_branch_from_resized_boxes  # noqa: E402
from utils.pln_candidate_pairs import generate_candidate_pairs_from_links  # noqa: E402
from utils.pln_pair_confidence import attach_pair_scores_and_labels_max_n  # noqa: E402
from utils.inference_geometry import uv_offset_to_image_xy, box_from_corner_and_center  # noqa: E402
from utils.nms import class_aware_nms  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("PLN small-set overfit check")
    p.add_argument("--voc2007_root", type=str, default=str(ROOT / "VOC2007" / "VOCdevkit" / "VOC2007"))
    p.add_argument("--voc2012_root", type=str, default=str(ROOT / "VOC2012" / "VOCdevkit" / "VOC2012"))
    p.add_argument("--use_voc2012", action="store_true")
    p.add_argument("--split", type=str, default="trainval")
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--subset_size", type=int, default=8, help="Number of images in tiny overfit set")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.00004)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone during overfit test")
    p.add_argument("--no_augment", action="store_true", help="Disable augmentation for easier overfit")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--output_dir", type=str, default=str(ROOT / "debug_out" / "overfit_smallset"))
    p.add_argument("--viz_every", type=int, default=20)
    p.add_argument("--viz_topk", type=int, default=60)
    p.add_argument("--conf_thres", type=float, default=0.25)
    p.add_argument("--nms_iou_threshold", type=float, default=0.45)
    p.add_argument("--nms_max_dets", type=int, default=60)
    p.add_argument(
        "--loss_input",
        type=str,
        default="logits",
        choices=["logits", "normalized"],
        help="Loss input type: raw logits or normalized predictions.",
    )
    p.add_argument(
        "--loss_reduction",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Loss reduction mode passed to PLNLoss.",
    )
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_targets_batch(
    targets_list: List[Dict],
    *,
    branch: str,
    image_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    B = len(targets_list)
    P_hat = torch.zeros((B, 4, 14, 14), device=device)
    Q_hat = torch.zeros((B, 4, 20, 14, 14), device=device)
    x_hat = torch.zeros((B, 4, 14, 14), device=device)
    y_hat = torch.zeros((B, 4, 14, 14), device=device)
    Lx_hat = torch.zeros((B, 4, 14, 14, 14), device=device)
    Ly_hat = torch.zeros((B, 4, 14, 14, 14), device=device)
    pt_mask = torch.zeros((B, 4, 14, 14), device=device)
    nopt_mask = torch.ones((B, 4, 14, 14), device=device)

    for bi in range(B):
        t = targets_list[bi]
        boxes = t["boxes"].to(device=device, dtype=torch.float32)
        labels = t["labels"].to(device=device, dtype=torch.int64)
        built = build_pln_targets_for_branch_from_resized_boxes(
            boxes_xyxy_resized=boxes,
            labels_idx=labels,
            branch=branch,  # type: ignore[arg-type]
            image_size=image_size,
            grid_size=14,
            B_point=2,
            device=device,
        )
        P_hat[bi] = built["P"]
        Q_hat[bi] = built["Q"]
        x_hat[bi] = built["x"]
        y_hat[bi] = built["y"]
        Lx_hat[bi] = built["Lx"]
        Ly_hat[bi] = built["Ly"]
        pt_mask[bi] = built["pt_mask"]
        nopt_mask[bi] = built["nopt_mask"]

    return {
        "P": P_hat,
        "Q": Q_hat,
        "x": x_hat,
        "y": y_hat,
        "Lx": Lx_hat,
        "Ly": Ly_hat,
        "pt_mask": pt_mask,
        "nopt_mask": nopt_mask,
    }


def _boxes_from_center_corner(center_img_xy: torch.Tensor, corner_img_xy: torch.Tensor) -> torch.Tensor:
    x_min, y_min, x_max, y_max = box_from_corner_and_center(corner_img_xy, center_img_xy)
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=torch.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = torch.maximum(ax1[:, None], bx1[None, :])
    iy1 = torch.maximum(ay1[:, None], by1[None, :])
    ix2 = torch.minimum(ax2[:, None], bx2[None, :])
    iy2 = torch.minimum(ay2[:, None], by2[None, :])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_a = ((ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0))[:, None]
    area_b = ((bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0))[None, :]
    return inter / (area_a + area_b - inter + 1e-6)


def _voc07_ap(rec: torch.Tensor, prec: torch.Tensor) -> float:
    ap = 0.0
    for t in torch.arange(0.0, 1.1, 0.1, device=rec.device):
        p = prec[rec >= t].max() if (rec >= t).any() else torch.tensor(0.0, device=rec.device)
        ap += float(p.item()) / 11.0
    return ap


def _map50_single_image(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes=20, iou_thres=0.5) -> float:
    present = gt_labels.unique()
    present = present[(present >= 0) & (present < num_classes)]
    if present.numel() == 0:
        return 0.0
    aps = []
    for c in present.tolist():
        gt_c = gt_boxes[gt_labels == c]
        pr_c = pred_boxes[pred_labels == c]
        sc_c = pred_scores[pred_labels == c]
        if gt_c.numel() == 0:
            continue
        if pr_c.numel() == 0:
            aps.append(0.0)
            continue
        order = torch.argsort(sc_c, descending=True)
        pr_c = pr_c[order]
        ious = _box_iou_xyxy(pr_c, gt_c)
        matched = torch.zeros((gt_c.shape[0],), device=gt_c.device, dtype=torch.bool)
        tp = torch.zeros((pr_c.shape[0],), device=gt_c.device)
        fp = torch.zeros((pr_c.shape[0],), device=gt_c.device)
        for i in range(pr_c.shape[0]):
            best_iou, best_j = ious[i].max(dim=0)
            if float(best_iou.item()) >= iou_thres and not matched[int(best_j.item())]:
                tp[i] = 1.0
                matched[int(best_j.item())] = True
            else:
                fp[i] = 1.0
        rec = torch.cumsum(tp, dim=0) / float(gt_c.shape[0])
        prec = torch.cumsum(tp, dim=0) / (torch.cumsum(tp, dim=0) + torch.cumsum(fp, dim=0) + 1e-6)
        aps.append(_voc07_ap(rec, prec))
    return float(sum(aps) / len(aps)) if aps else 0.0


@torch.no_grad()
def _save_overfit_viz(
    *,
    out_dir: Path,
    step: int,
    images: torch.Tensor,
    targets_list: List[Dict],
    outs: Dict[str, torch.Tensor],
    loss_value: float,
    loss_breakdown: dict | None,
    suggested_weights: dict | None,
    viz_topk: int,
    conf_thres: float,
    nms_iou_threshold: float,
    nms_max_dets: int,
    image_size: int,
) -> None:
    def _fmt_ratio_dict(d: dict | None) -> str:
        if not d:
            return "-"
        keys = ["p", "q", "coord", "link", "nopt"]
        parts = []
        for k in keys:
            if k in d:
                parts.append(f"{k}:{float(d[k]):.3f}")
        return "{" + ", ".join(parts) + "}"

    def _fmt_w_dict(d: dict | None) -> str:
        if not d:
            return "-"
        keys = ["w_class", "w_coord", "w_link"]
        parts = []
        for k in keys:
            if k in d:
                parts.append(f"{k}:{float(d[k]):.2e}")
        return "{" + ", ".join(parts) + "}"

    S = 14
    stride = image_size // S
    b = 0
    img_t = images[b].detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img_vis = (img_t * std + mean).clamp(0, 1)

    boxes_all, scores_all, labels_all = [], [], []
    score_stats = {"max": 0.0, "mean": 0.0}
    score_topk: list[float] = []
    branch_pair_counts: dict[str, int] = {}
    for branch in ["left_top", "right_top", "left_bottom", "right_bottom"]:
        dec = decode_branch_channels_inference(outs[branch], s=S)
        pairs = generate_candidate_pairs_from_links(dec["P"], dec["Lx"], dec["Ly"], branch=branch, s=S, p_threshold=0.0)
        branch_pair_counts[branch] = int(pairs["pairs_ijst"].shape[0])
        if pairs["pairs_ijst"].shape[0] == 0:
            continue
        pairs = attach_pair_scores_and_labels_max_n(P=dec["P"], Q=dec["Q"], Lx=dec["Lx"], Ly=dec["Ly"], pairs_dict=pairs)

        # score diagnostics (before threshold)
        sc_all = pairs["score"].detach()
        if sc_all.numel() > 0:
            score_stats["max"] = max(score_stats["max"], float(sc_all.max().item()))
            score_stats["mean"] += float(sc_all.mean().item())
            k = min(10, int(sc_all.numel()))
            score_topk.extend([float(x) for x in torch.topk(sc_all, k=k).values.cpu().tolist()])

        keep = pairs["score"] >= conf_thres if conf_thres > 0 else torch.ones_like(pairs["score"], dtype=torch.bool)
        # If threshold removes everything, keep top-k for visualization (so we can still inspect training).
        if keep.sum().item() == 0:
            # select top k
            k = min(max(1, viz_topk), int(pairs["score"].numel()))
            if k == 0:
                continue
            sel = torch.topk(pairs["score"], k=k).indices
            for kname in ["score", "label", "center_xy", "corner_xy", "center_point_type", "corner_point_type", "batch_idx"]:
                pairs[kname] = pairs[kname][sel]
        else:
            for kname in ["score", "label", "center_xy", "corner_xy", "center_point_type", "corner_point_type", "batch_idx"]:
                pairs[kname] = pairs[kname][keep]

        # Apply top-K before per-image filter/NMS
        if viz_topk > 0 and pairs["score"].numel() > viz_topk:
            sel = torch.topk(pairs["score"], k=viz_topk).indices
            for kname in ["score", "label", "center_xy", "corner_xy", "center_point_type", "corner_point_type", "batch_idx"]:
                pairs[kname] = pairs[kname][sel]
        m = pairs["batch_idx"] == b
        if m.sum().item() == 0:
            continue
        score = pairs["score"][m]
        lab = pairs["label"][m]
        center_p = pairs["center_point_type"][m]
        corner_p = pairs["corner_point_type"][m]
        center_u = pairs["center_xy"][m][:, 0].round().to(torch.long)
        center_v = pairs["center_xy"][m][:, 1].round().to(torch.long)
        corner_u = pairs["corner_xy"][m][:, 0].round().to(torch.long)
        corner_v = pairs["corner_xy"][m][:, 1].round().to(torch.long)
        center_dx = dec["dx"][b, center_p, center_v, center_u]
        center_dy = dec["dy"][b, center_p, center_v, center_u]
        corner_dx = dec["dx"][b, corner_p, corner_v, corner_u]
        corner_dy = dec["dy"][b, corner_p, corner_v, corner_u]
        cx, cy = uv_offset_to_image_xy(u=center_u, v=center_v, dx=center_dx, dy=center_dy, stride=stride)
        kx, ky = uv_offset_to_image_xy(u=corner_u, v=corner_v, dx=corner_dx, dy=corner_dy, stride=stride)
        boxes = _boxes_from_center_corner(torch.stack([cx, cy], dim=1), torch.stack([kx, ky], dim=1))
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(image_size))
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(image_size))
        boxes_all.append(boxes.detach().cpu())
        scores_all.append(score.detach().cpu())
        labels_all.append(lab.detach().cpu())

    if boxes_all:
        pred_boxes = torch.cat(boxes_all, dim=0)
        pred_scores = torch.cat(scores_all, dim=0)
        pred_labels = torch.cat(labels_all, dim=0)
        keep_idx = class_aware_nms(pred_boxes, pred_scores, pred_labels, iou_threshold=nms_iou_threshold, max_dets=nms_max_dets)
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep_idx], pred_scores[keep_idx], pred_labels[keep_idx]
    else:
        pred_boxes = torch.zeros((0, 4))
        pred_scores = torch.zeros((0,))
        pred_labels = torch.zeros((0,), dtype=torch.long)

    # Final cap for visualization: keep top viz_topk AFTER NMS (global, per-image).
    if viz_topk > 0 and pred_scores.numel() > viz_topk:
        sel = torch.topk(pred_scores, k=viz_topk).indices
        pred_boxes = pred_boxes[sel]
        pred_scores = pred_scores[sel]
        pred_labels = pred_labels[sel]

    gt_boxes = targets_list[b]["boxes"].detach().cpu()
    gt_labels = targets_list[b]["labels"].detach().cpu()
    map50 = _map50_single_image(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.imshow(img_vis.permute(1, 2, 0).numpy())
    ax.axis("off")
    score_topk = sorted(score_topk, reverse=True)[:3]
    topk_text = "[" + ", ".join(f"{v:.2e}" for v in score_topk) + "]"
    ratio_text = _fmt_ratio_dict(loss_breakdown)
    w_text = _fmt_w_dict(suggested_weights)
    ax.set_title(
        f"step={step} | loss={loss_value:.4f} | mAP50={map50:.4f}\n"
        f"conf_thres={conf_thres} | viz_topk={viz_topk} | pred_after_nms={int(pred_scores.numel())}\n"
        f"score_max={score_stats['max']:.2e} | top3={topk_text}\n"
        f"loss_ratio={ratio_text}\n"
        f"w_eq={w_text}"
    )
    for bb, ll in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = [float(v) for v in bb.tolist()]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        ax.text(x1, max(0.0, y1 - 2), f"GT:{int(ll)}", color="black", fontsize=8, backgroundcolor="lime")
    for bb, ll, sc in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = [float(v) for v in bb.tolist()]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
        ax.text(x1, max(0.0, y1 - 2), f"P:{int(ll)} {float(sc):.3f}", color="yellow", fontsize=8, backgroundcolor="black")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"overfit_viz_step_{step:06d}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    if args.gpu_id >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but --gpu_id was set.")
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_2007 = VOC2007Dataset(
        voc_root_dir=args.voc2007_root,
        split=args.split,
        output_size=(args.image_size, args.image_size),
        augment=not args.no_augment,
        dataset_tag="2007",
    )
    if args.use_voc2012:
        ds_2012 = VOC2007Dataset(
            voc_root_dir=args.voc2012_root,
            split=args.split,
            output_size=(args.image_size, args.image_size),
            augment=not args.no_augment,
            dataset_tag="2012",
        )
        from torch.utils.data import ConcatDataset

        ds_all = ConcatDataset([ds_2007, ds_2012])
    else:
        ds_all = ds_2007

    subset_size = min(args.subset_size, len(ds_all))
    indices = list(range(subset_size))
    ds_small = Subset(ds_all, indices)

    dl = DataLoader(
        ds_small,
        batch_size=min(args.batch_size, subset_size),
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_voc_detection,
        drop_last=True,
    )
    if len(dl) == 0:
        raise RuntimeError("DataLoader has 0 batches. Increase subset_size or reduce batch_size.")

    model = PLNModel(backbone_pretrained=False, backbone_trainable=not args.freeze_backbone, freeze_bn=False).to(device)
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = PLNLoss(
        weights=PLNLossWeights(w_class=1.0, w_coord=1.0, w_link=1.0),
        reduction=args.loss_reduction,
    )

    model.train()
    losses: List[float] = []
    branch_keys = ["left_top", "right_top", "left_bottom", "right_bottom"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = run_dir / "train_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    dl_iter = iter(dl)
    pbar = tqdm(range(1, args.steps + 1), desc="Overfit tiny-set", dynamic_ncols=True)
    for step in pbar:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        images = batch["images"].to(device)
        targets_list = batch["targets"]

        outs = model(images)
        loss_total = 0.0
        loss_nopt_total = 0.0
        loss_pt_p_total = 0.0
        loss_pt_qw_total = 0.0
        loss_pt_coordw_total = 0.0
        loss_pt_linkw_total = 0.0
        loss_pt_qraw_total = 0.0
        loss_pt_coordraw_total = 0.0
        loss_pt_linkraw_total = 0.0
        for branch in branch_keys:
            pred_map = outs[branch]
            if args.loss_input == "normalized":
                pred_dec = decode_branch_channels_inference(pred_map, s=14, num_classes=20, num_points=4)
                pred_for_loss = {
                    "P": pred_dec["P"],
                    "Q": pred_dec["Q"],
                    "x": pred_dec["dx"],
                    "y": pred_dec["dy"],
                    "Lx": pred_dec["Lx"],
                    "Ly": pred_dec["Ly"],
                }
            else:
                pred_dec = decode_branch_channels_logits(pred_map, s=14, num_classes=20, num_points=4)
                pred_for_loss = {
                    "P": pred_dec["P_logits"],
                    "Q": pred_dec["Q_logits"],
                    "x": pred_dec["dx_logits"],
                    "y": pred_dec["dy_logits"],
                    "Lx": pred_dec["Lx_logits"],
                    "Ly": pred_dec["Ly_logits"],
                }
            tgt = _build_targets_batch(targets_list, branch=branch, image_size=args.image_size, device=device)

            loss_dict = criterion(
                pred=pred_for_loss,
                target={
                    "P": tgt["P"],
                    "Q": tgt["Q"],
                    "x": tgt["x"],
                    "y": tgt["y"],
                    "Lx": tgt["Lx"],
                    "Ly": tgt["Ly"],
                },
                pt_mask=tgt["pt_mask"],
                nopt_mask=tgt["nopt_mask"],
            )
            loss_total = loss_total + loss_dict["loss"]
            loss_nopt_total = loss_nopt_total + loss_dict["loss_nopt"]
            loss_pt_p_total = loss_pt_p_total + loss_dict["loss_pt_p"]
            loss_pt_qw_total = loss_pt_qw_total + loss_dict["loss_pt_q_weighted"]
            loss_pt_coordw_total = loss_pt_coordw_total + loss_dict["loss_pt_coord_weighted"]
            loss_pt_linkw_total = loss_pt_linkw_total + loss_dict["loss_pt_link_weighted"]
            loss_pt_qraw_total = loss_pt_qraw_total + loss_dict["loss_pt_q_raw"]
            loss_pt_coordraw_total = loss_pt_coordraw_total + loss_dict["loss_pt_coord_raw"]
            loss_pt_linkraw_total = loss_pt_linkraw_total + loss_dict["loss_pt_link_raw"]

        loss = loss_total / 4.0
        loss_nopt_avg = loss_nopt_total / 4.0
        loss_pt_p_avg = loss_pt_p_total / 4.0
        loss_pt_qw_avg = loss_pt_qw_total / 4.0
        loss_pt_coordw_avg = loss_pt_coordw_total / 4.0
        loss_pt_linkw_avg = loss_pt_linkw_total / 4.0
        loss_pt_qraw_avg = loss_pt_qraw_total / 4.0
        loss_pt_coordraw_avg = loss_pt_coordraw_total / 4.0
        loss_pt_linkraw_avg = loss_pt_linkraw_total / 4.0
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        losses.append(loss_val)
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})
        if step % args.log_every == 0:
            print(f"[step {step:04d}/{args.steps}] loss={loss_val:.6f}")
        total_now = float(loss.item()) + 1e-8
        loss_breakdown = {
            "p": float(loss_pt_p_avg.item()) / total_now,
            "q": float(loss_pt_qw_avg.item()) / total_now,
            "coord": float(loss_pt_coordw_avg.item()) / total_now,
            "link": float(loss_pt_linkw_avg.item()) / total_now,
            "nopt": float(loss_nopt_avg.item()) / total_now,
        }
        p_base = max(float(loss_pt_p_avg.item()), 1e-8)
        suggested_weights = {
            "w_class": p_base / max(float(loss_pt_qraw_avg.item()), 1e-8),
            "w_coord": p_base / max(float(loss_pt_coordraw_avg.item()), 1e-8),
            "w_link": p_base / max(float(loss_pt_linkraw_avg.item()), 1e-8),
        }
        if args.viz_every > 0 and (step % args.viz_every == 0):
            _save_overfit_viz(
                out_dir=viz_dir,
                step=step,
                images=images,
                targets_list=targets_list,
                outs=outs,
                loss_value=loss_val,
                loss_breakdown=loss_breakdown,
                suggested_weights=suggested_weights,
                viz_topk=args.viz_topk,
                conf_thres=args.conf_thres,
                nms_iou_threshold=args.nms_iou_threshold,
                nms_max_dets=args.nms_max_dets,
                image_size=args.image_size,
            )

    # Save JSON summary
    first_loss = losses[0]
    last_loss = losses[-1]
    min_loss = min(losses)
    drop_ratio = (first_loss - last_loss) / max(first_loss, 1e-8)
    summary = {
        "config": vars(args),
        "device": str(device),
        "subset_size_actual": subset_size,
        "num_steps": args.steps,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "min_loss": min_loss,
        "loss_drop_ratio": drop_ratio,
        "is_overfit_trending": bool(last_loss < first_loss),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "loss_curve.json", "w") as f:
        json.dump({"loss": losses}, f, indent=2)

    # Save loss plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, linewidth=1.5)
    ax.set_title("Tiny-set overfit loss curve")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # Save checkpoint
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "summary": summary,
        },
        run_dir / "overfit_model_final.pth",
    )

    print("\n=== Overfit tiny-set result ===")
    print(f"run_dir: {run_dir}")
    print(f"first_loss={first_loss:.6f}")
    print(f"last_loss={last_loss:.6f}")
    print(f"min_loss={min_loss:.6f}")
    print(f"loss_drop_ratio={drop_ratio:.4f}")
    print(f"is_overfit_trending={summary['is_overfit_trending']}")


if __name__ == "__main__":
    main()

