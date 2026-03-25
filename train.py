from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.pln_model import PLNModel  # noqa: E402
from datasets.image_dataset import ImageOnlyDataset  # noqa: E402
from datasets.collate import collate_images  # noqa: E402
from datasets.voc_dataset import VOC2007Dataset  # noqa: E402
from datasets.collate_detection import collate_voc_detection  # noqa: E402

from utils.pln_channel_decoder import decode_branch_channels_inference, decode_branch_channels_logits  # noqa: E402
from utils.pln_loss import PLNLoss, PLNLossWeights  # noqa: E402
from utils.pln_target_builder import build_pln_targets_for_branch_from_resized_boxes  # noqa: E402
from utils.pln_target_builder_gaussian_links import build_pln_targets_for_branch_from_resized_boxes_gaussian_links  # noqa: E402
from utils.pln_candidate_pairs import generate_candidate_pairs_from_links  # noqa: E402
from utils.pln_pair_confidence import attach_pair_scores_and_labels_max_n  # noqa: E402
from utils.inference_geometry import uv_offset_to_image_xy, box_from_corner_and_center  # noqa: E402
from utils.nms import class_aware_nms  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("PLN Train (skeleton)")
    p.add_argument(
        "--dataset_type",
        type=str,
        default="voc",
        choices=["voc", "image_only"],
        help="Which dataset loader to use.",
    )

    p.add_argument(
        "--data_root",
        type=str,
        default="",
        help="For dataset_type=image_only: dataset root containing split dirs.",
    )
    default_voc2007_root = str(ROOT / "VOC2007" / "VOCdevkit" / "VOC2007")
    p.add_argument(
        "--voc2007_root",
        type=str,
        default=default_voc2007_root,
        help="For dataset_type=voc: path to VOCdevkit/VOC2007",
    )
    default_voc2012_root = str(ROOT / "VOC2012" / "VOCdevkit" / "VOC2012")
    p.add_argument(
        "--voc2012_root",
        type=str,
        default=default_voc2012_root,
        help="For dataset_type=voc: path to VOCdevkit/VOC2012",
    )
    p.add_argument("--split", type=str, default="trainval", help="VOC split: trainval/train/val/test")

    p.add_argument("--image_size", type=int, default=448, help="Final square size (H=W=image_size)")
    p.add_argument("--batch_size", type=int, default=64)

    # Learning rate schedule:
    # - start at 0.001
    # - linearly increase to 0.005 over warmup_iters
    # - then keep 0.005 for finetune_iters iterations
    p.add_argument("--lr_start", type=float, default=0.001)
    p.add_argument("--lr_end", type=float, default=0.005)
    p.add_argument("--warmup_iters", type=int, default=10000, help="Warmup iterations to reach lr_end")
    p.add_argument("--finetune_iters", type=int, default=30000, help="Iterations to train at lr_end")

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="Optimizer type. Default keeps previous behavior (sgd).",
    )
    p.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer.")
    p.add_argument("--weight_decay", type=float, default=0.00004, help="Weight decay for optimizer.")
    p.add_argument("--pretrained_backbone", action="store_true", help="Use pretrained ResNet18 weights")
    p.add_argument(
        "--train_backbone",
        action="store_true",
        help="If set, backbone parameters are trainable (can still be initialized by --pretrained_backbone).",
    )
    p.add_argument("--no_freeze_bn", action="store_true", help="Keep BN trainable (default: freeze BN)")
    p.add_argument(
        "--no_voc2012_train",
        action="store_true",
        help="Disable VOC2012trainval from training (default: enabled).",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--viz_every", type=int, default=10, help="Save one train visualization every N iterations")
    p.add_argument("--viz_topk", type=int, default=60, help="For visualization: keep top-K scored pairs (after optional conf_thres) before NMS")
    p.add_argument("--viz_topk_after_nms", type=int, default=60, help="For visualization: final top-K cap after NMS; set <=0 to disable")
    p.add_argument("--conf_thres", type=float, default=0.25, help="Train-time inference score threshold")
    p.add_argument("--nms_iou_threshold", type=float, default=0.45)
    p.add_argument("--nms_max_dets", type=int, default=60)
    p.add_argument("--gpu_id", type=int, default=-1, help="CUDA device index. -1 means keep default device setting.")
    p.add_argument("--save_every", type=int, default=1000, help="Save periodic checkpoint every N iterations")
    p.add_argument("--max_grad_norm", type=float, default=5.0, help="Gradient clipping norm; <=0 disables")
    p.add_argument("--loss_logit_clip", type=float, default=20.0, help="Clip raw logits in loss for numerical stability; <=0 disables")
    p.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path to load model weights from")
    p.add_argument(
        "--loss_input",
        type=str,
        default="logits",
        choices=["logits", "normalized"],
        help="Loss input type: raw logits or normalized predictions.",
    )

    # Link target smoothing (gaussian neighborhood on Lx/Ly)
    p.add_argument(
        "--use_gaussian_link_targets",
        action="store_true",
        help="Use gaussian neighborhood supervision for Lx/Ly link targets (instead of hard one-hot).",
    )
    p.add_argument("--gaussian_link_sigma", type=float, default=1.0, help="Gaussian sigma for link targets.")
    p.add_argument(
        "--gaussian_link_radius",
        type=int,
        default=1,
        help="Gaussian radius (in k steps) for link targets (0 keeps hard one-hot).",
    )
    return p.parse_args()


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
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    area_a = ((ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0))[:, None]
    area_b = ((bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0))[None, :]
    union = (area_a + area_b - inter).clamp(min=1e-6)
    return inter / union


def _voc07_ap(rec: torch.Tensor, prec: torch.Tensor) -> float:
    ap = 0.0
    for t in torch.arange(0.0, 1.1, 0.1, device=rec.device):
        p = prec[rec >= t].max() if (rec >= t).any() else torch.tensor(0.0, device=rec.device)
        ap += float(p.item()) / 11.0
    return ap


def _map50_single_image(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    *,
    num_classes: int = 20,
    iou_thres: float = 0.5,
) -> float:
    present = gt_labels.unique()
    present = present[(present >= 0) & (present < num_classes)]
    if present.numel() == 0:
        return 0.0

    aps: list[float] = []
    for c in present.tolist():
        gt_mask = gt_labels == c
        pr_mask = pred_labels == c
        gt_c = gt_boxes[gt_mask]
        pr_c = pred_boxes[pr_mask]
        sc_c = pred_scores[pr_mask]
        if gt_c.numel() == 0:
            continue
        if pr_c.numel() == 0:
            aps.append(0.0)
            continue

        order = torch.argsort(sc_c, descending=True)
        pr_c = pr_c[order]
        ious = _box_iou_xyxy(pr_c, gt_c)
        matched = torch.zeros((gt_c.shape[0],), device=gt_c.device, dtype=torch.bool)
        tp = torch.zeros((pr_c.shape[0],), device=gt_c.device, dtype=torch.float32)
        fp = torch.zeros((pr_c.shape[0],), device=gt_c.device, dtype=torch.float32)

        for i in range(pr_c.shape[0]):
            best_iou, best_j = ious[i].max(dim=0)
            if float(best_iou.item()) >= iou_thres and not matched[int(best_j.item())]:
                tp[i] = 1.0
                matched[int(best_j.item())] = True
            else:
                fp[i] = 1.0

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        rec = tp_cum / float(gt_c.shape[0])
        prec = tp_cum / (tp_cum + fp_cum + 1e-6)
        aps.append(_voc07_ap(rec, prec))

    if not aps:
        return 0.0
    return float(sum(aps) / len(aps))


@torch.no_grad()
def _save_train_viz(
    *,
    out_dir: Path,
    it: int,
    images: torch.Tensor,
    targets_list: list[dict],
    outs: dict,
    loss_value: float,
    loss_breakdown: dict | None,
    suggested_weights: dict | None,
    viz_topk: int,
    viz_topk_after_nms: int,
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

    boxes_all: list[torch.Tensor] = []
    scores_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    score_max_global = 0.0
    score_topk_global: list[float] = []
    branch_pair_counts: dict[str, int] = {}

    for branch in ["left_top", "right_top", "left_bottom", "right_bottom"]:
        feat = outs[branch]
        dec = decode_branch_channels_inference(feat, s=S)
        pairs = generate_candidate_pairs_from_links(dec["P"], dec["Lx"], dec["Ly"], branch=branch, s=S, p_threshold=0.0)
        branch_pair_counts[branch] = int(pairs["pairs_ijst"].shape[0])
        if pairs["pairs_ijst"].shape[0] == 0:
            continue
        pairs = attach_pair_scores_and_labels_max_n(P=dec["P"], Q=dec["Q"], Lx=dec["Lx"], Ly=dec["Ly"], pairs_dict=pairs)

        score = pairs["score"]
        lab = pairs["label"]
        if score.numel() > 0:
            score_max_global = max(score_max_global, float(score.max().item()))
            k0 = min(10, int(score.numel()))
            score_topk_global.extend([float(x) for x in torch.topk(score, k=k0).values.cpu().tolist()])
        if conf_thres > 0:
            keep = score >= conf_thres
            if keep.sum().item() == 0:
                # Keep top-k for visualization even if under threshold, so user can inspect.
                k = min(max(1, viz_topk), int(score.numel()))
                if k == 0:
                    continue
                sel = torch.topk(score, k=k).indices
                score = score[sel]
                lab = lab[sel]
                for kname in [
                    "center_xy",
                    "corner_xy",
                    "center_point_type",
                    "corner_point_type",
                    "batch_idx",
                ]:
                    pairs[kname] = pairs[kname][sel]
            else:
                score = score[keep]
                lab = lab[keep]
                for kname in [
                    "center_xy",
                    "corner_xy",
                    "center_point_type",
                    "corner_point_type",
                    "batch_idx",
                ]:
                    pairs[kname] = pairs[kname][keep]
        # Apply top-K for visualization before per-image filtering/NMS
        if viz_topk > 0 and score.numel() > viz_topk:
            sel = torch.topk(score, k=viz_topk).indices
            score = score[sel]
            lab = lab[sel]
            for kname in [
                "center_xy",
                "corner_xy",
                "center_point_type",
                "corner_point_type",
                "batch_idx",
            ]:
                pairs[kname] = pairs[kname][sel]

        m = pairs["batch_idx"] == b
        if m.sum().item() == 0:
            continue
        score = score[m]
        lab = lab[m]
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

        center_x_img, center_y_img = uv_offset_to_image_xy(u=center_u, v=center_v, dx=center_dx, dy=center_dy, stride=stride)
        corner_x_img, corner_y_img = uv_offset_to_image_xy(u=corner_u, v=corner_v, dx=corner_dx, dy=corner_dy, stride=stride)
        boxes = _boxes_from_center_corner(
            center_img_xy=torch.stack([center_x_img, center_y_img], dim=1),
            corner_img_xy=torch.stack([corner_x_img, corner_y_img], dim=1),
        )
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(image_size))
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(image_size))

        boxes_all.append(boxes.detach().cpu())
        scores_all.append(score.detach().cpu())
        labels_all.append(lab.detach().cpu())

    if boxes_all:
        pred_boxes = torch.cat(boxes_all, dim=0)
        pred_scores = torch.cat(scores_all, dim=0)
        pred_labels = torch.cat(labels_all, dim=0)
        keep_idx = class_aware_nms(
            pred_boxes,
            pred_scores,
            pred_labels,
            iou_threshold=nms_iou_threshold,
            max_dets=nms_max_dets,
        )
        pred_boxes = pred_boxes[keep_idx]
        pred_scores = pred_scores[keep_idx]
        pred_labels = pred_labels[keep_idx]
    else:
        pred_boxes = torch.zeros((0, 4))
        pred_scores = torch.zeros((0,))
        pred_labels = torch.zeros((0,), dtype=torch.long)

    # Optional final cap for visualization after NMS (global, per-image).
    if viz_topk_after_nms > 0 and pred_scores.numel() > viz_topk_after_nms:
        sel = torch.topk(pred_scores, k=viz_topk_after_nms).indices
        pred_boxes = pred_boxes[sel]
        pred_scores = pred_scores[sel]
        pred_labels = pred_labels[sel]

    gt_boxes = targets_list[b]["boxes"].detach().cpu()
    gt_labels = targets_list[b]["labels"].detach().cpu()
    map50 = _map50_single_image(
        pred_boxes,
        pred_scores,
        pred_labels,
        gt_boxes,
        gt_labels,
        num_classes=20,
        iou_thres=0.5,
    )

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.imshow(img_vis.permute(1, 2, 0).numpy())
    ax.axis("off")
    score_topk_global = sorted(score_topk_global, reverse=True)[:3]
    topk_text = "[" + ", ".join(f"{v:.2e}" for v in score_topk_global) + "]"
    ratio_text = _fmt_ratio_dict(loss_breakdown)
    w_text = _fmt_w_dict(suggested_weights)
    ax.set_title(
        f"iter={it} | loss={loss_value:.4f} | mAP50={map50:.4f}\n"
        f"conf_thres={conf_thres} | viz_topk={viz_topk} | viz_topk_after_nms={viz_topk_after_nms} | pred_after_nms={int(pred_scores.numel())}\n"
        f"score_max={score_max_global:.2e} | top3={topk_text}\n"
        f"loss_ratio={ratio_text}\n"
        f"w_eq={w_text}"
    )

    for bb, ll in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = [float(x) for x in bb.tolist()]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        ax.text(x1, max(0.0, y1 - 2), f"GT:{int(ll)}", color="black", fontsize=8, backgroundcolor="lime")

    for bb, ll, sc in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = [float(x) for x in bb.tolist()]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
        ax.text(x1, max(0.0, y1 - 2), f"P:{int(ll)} {float(sc):.3f}", color="yellow", fontsize=8, backgroundcolor="black")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"train_viz_iter_{it:06d}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.gpu_id >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but --gpu_id was specified.")
        args.device = f"cuda:{args.gpu_id}"

    # NOTE: This is still a scaffolding trainer. The purpose here is:
    # - use VOC2007 dataloader + required augmentation
    # - apply the SGD hyperparameters + iteration-wise LR schedule
    # - keep a dummy loss until the detection head / loss are implemented.
    transform = Compose(
        [
            Resize((args.image_size, args.image_size), antialias=True),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if args.dataset_type == "voc":
        # Training uses VOC2007 + VOC2012 (VOC2007 test only later).
        ds_2007 = VOC2007Dataset(
            voc_root_dir=args.voc2007_root,
            split=args.split,
            output_size=(args.image_size, args.image_size),
            augment=True,
            dataset_tag="2007",
        )

        if not args.no_voc2012_train:
            
            ds_2012 = VOC2007Dataset(
                voc_root_dir=args.voc2012_root,
                split=args.split,
                output_size=(args.image_size, args.image_size),
                augment=True,
                dataset_tag="2012",
            )
            from torch.utils.data import ConcatDataset

            ds = ConcatDataset([ds_2007, ds_2012])
        else:
            ds = ds_2007

        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_voc_detection,
            drop_last=True,
        )
    else:
        ds = ImageOnlyDataset(root_dir=args.data_root, split_dir=args.split, transform=transform)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_images,
            drop_last=True,
        )

    model = PLNModel(
        backbone_pretrained=args.pretrained_backbone,
        backbone_trainable=args.train_backbone,
        freeze_bn=not args.no_freeze_bn,
    ).to(args.device)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)

    # Keep default behavior (frozen backbone) unless --train_backbone is set.
    if not args.train_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adam":
        optimizer = Adam(
            trainable_params,
            lr=args.lr_start,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = SGD(
            trainable_params,
            lr=args.lr_start,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    criterion = PLNLoss(
        weights=PLNLossWeights(w_class=1.0, w_coord=1.0, w_link=1.0),
        reduction="mean",
        logit_clip=(args.loss_logit_clip if args.loss_logit_clip > 0 else None),
    )

    model.train()

    total_iters = args.warmup_iters + args.finetune_iters
    it = 0
    running = 0.0
    base_out_dir = Path(args.voc2007_root if args.dataset_type == "voc" else args.data_root).parent / "pln_runs"
    base_out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = run_dir / "train_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    # Save run config
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Setup logger (stdout + file)
    logger = logging.getLogger("pln_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)

    logger.info("Run directory: %s", run_dir)
    logger.info("TensorBoard: tensorboard --logdir=%s", tb_dir)
    with open(base_out_dir / "log.txt", "w") as f:
        f.write(f"tensorboard --logdir={tb_dir}\n")
    logger.info("Config saved: %s", config_path)

    dl_iter = iter(dl)
    pbar = tqdm(total=total_iters, desc="PLN Train", dynamic_ncols=True)
    while it < total_iters:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        if args.dataset_type == "voc":
            images = batch["images"].to(args.device)
            # targets are available as batch["targets"], but dummy loss ignores them for now.
        else:
            images = batch["image"].to(args.device)

        # Iteration-wise LR update (linear warmup).
        if it < args.warmup_iters:
            t = it / float(max(1, args.warmup_iters))
            lr = args.lr_start + (args.lr_end - args.lr_start) * t
        else:
            lr = args.lr_end

        for group in optimizer.param_groups:
            group["lr"] = lr

        outs = model(images)
        if not isinstance(outs, dict):
            raise RuntimeError("PLNModel is expected to return a dict of 4 branches.")

        # Build per-branch targets from VOC resized boxes and compute loss.
        # Dataset boxes are in pixel coords after resize to (image_size,image_size).
        loss_total = 0.0
        loss_pt_total = 0.0
        loss_nopt_total = 0.0
        loss_pt_p_total = 0.0
        loss_pt_qw_total = 0.0
        loss_pt_coordw_total = 0.0
        loss_pt_linkw_total = 0.0
        loss_pt_qraw_total = 0.0
        loss_pt_coordraw_total = 0.0
        loss_pt_linkraw_total = 0.0

        if args.dataset_type != "voc":
            raise RuntimeError("This trainer currently supports loss only for VOC dataset_type.")

        targets_list = batch["targets"]
        branch_keys = ["left_top", "right_top", "left_bottom", "right_bottom"]

        for branch in branch_keys:
            pred_map = outs[branch]  # (B,204,14,14)
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

            # Build batch target tensors
            B = images.shape[0]
            P_hat = torch.zeros((B, 4, 14, 14), device=args.device)
            Q_hat = torch.zeros((B, 4, 20, 14, 14), device=args.device)
            x_hat = torch.zeros((B, 4, 14, 14), device=args.device)
            y_hat = torch.zeros((B, 4, 14, 14), device=args.device)
            Lx_hat = torch.zeros((B, 4, 14, 14, 14), device=args.device)  # (B,P,k,v,u)
            Ly_hat = torch.zeros((B, 4, 14, 14, 14), device=args.device)
            pt_mask = torch.zeros((B, 4, 14, 14), device=args.device)
            nopt_mask = torch.ones((B, 4, 14, 14), device=args.device)

            for bi in range(B):
                t = targets_list[bi]
                boxes = t["boxes"].to(device=args.device, dtype=torch.float32)
                labels = t["labels"].to(device=args.device, dtype=torch.int64)
                if args.use_gaussian_link_targets:
                    built = build_pln_targets_for_branch_from_resized_boxes_gaussian_links(
                        boxes_xyxy_resized=boxes,
                        labels_idx=labels,
                        branch=branch,  # type: ignore[arg-type]
                        image_size=args.image_size,
                        grid_size=14,
                        B_point=2,
                        gaussian_radius=args.gaussian_link_radius,
                        gaussian_sigma=args.gaussian_link_sigma,
                        device=torch.device(args.device),
                    )
                else:
                    built = build_pln_targets_for_branch_from_resized_boxes(
                        boxes_xyxy_resized=boxes,
                        labels_idx=labels,
                        branch=branch,  # type: ignore[arg-type]
                        image_size=args.image_size,
                        grid_size=14,
                        B_point=2,
                        device=torch.device(args.device),
                    )
                P_hat[bi] = built["P"]
                Q_hat[bi] = built["Q"]
                x_hat[bi] = built["x"]
                y_hat[bi] = built["y"]
                Lx_hat[bi] = built["Lx"]
                Ly_hat[bi] = built["Ly"]
                pt_mask[bi] = built["pt_mask"]
                nopt_mask[bi] = built["nopt_mask"]

            loss_dict = criterion(
                pred=pred_for_loss,
                target={
                    "P": P_hat,
                    "Q": Q_hat,
                    "x": x_hat,
                    "y": y_hat,
                    "Lx": Lx_hat,
                    "Ly": Ly_hat,
                },
                pt_mask=pt_mask,
                nopt_mask=nopt_mask,
            )

            loss_total = loss_total + loss_dict["loss"]
            loss_pt_total = loss_pt_total + loss_dict["loss_pt"]
            loss_nopt_total = loss_nopt_total + loss_dict["loss_nopt"]
            loss_pt_p_total = loss_pt_p_total + loss_dict["loss_pt_p"]
            loss_pt_qw_total = loss_pt_qw_total + loss_dict["loss_pt_q_weighted"]
            loss_pt_coordw_total = loss_pt_coordw_total + loss_dict["loss_pt_coord_weighted"]
            loss_pt_linkw_total = loss_pt_linkw_total + loss_dict["loss_pt_link_weighted"]
            loss_pt_qraw_total = loss_pt_qraw_total + loss_dict["loss_pt_q_raw"]
            loss_pt_coordraw_total = loss_pt_coordraw_total + loss_dict["loss_pt_coord_raw"]
            loss_pt_linkraw_total = loss_pt_linkraw_total + loss_dict["loss_pt_link_raw"]

        # Average over branches
        loss = loss_total / 4.0
        loss_pt_avg = loss_pt_total / 4.0
        loss_nopt_avg = loss_nopt_total / 4.0
        loss_pt_p_avg = loss_pt_p_total / 4.0
        loss_pt_qw_avg = loss_pt_qw_total / 4.0
        loss_pt_coordw_avg = loss_pt_coordw_total / 4.0
        loss_pt_linkw_avg = loss_pt_linkw_total / 4.0
        loss_pt_qraw_avg = loss_pt_qraw_total / 4.0
        loss_pt_coordraw_avg = loss_pt_coordraw_total / 4.0
        loss_pt_linkraw_avg = loss_pt_linkraw_total / 4.0

        if not torch.isfinite(loss):
            logger.error("Non-finite loss at iter=%d. Skip optimizer step.", it + 1)
            for bk in branch_keys:
                pm = outs[bk].detach()
                logger.error(
                    "  branch=%s pred_map stats: min=%.3e max=%.3e mean=%.3e",
                    bk,
                    float(pm.min().item()),
                    float(pm.max().item()),
                    float(pm.mean().item()),
                )
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
        optimizer.step()

        running += loss.item()
        it += 1
        pbar.update(1)
        pbar.set_postfix({"lr": f"{lr:.6f}", "loss": f"{loss.item():.4f}"})

        writer.add_scalar("train/loss", float(loss.item()), it)
        writer.add_scalar("train/loss_pt", float(loss_pt_avg.item()), it)
        writer.add_scalar("train/loss_nopt", float(loss_nopt_avg.item()), it)
        writer.add_scalar("train/loss_pt_p", float(loss_pt_p_avg.item()), it)
        writer.add_scalar("train/loss_pt_q_weighted", float(loss_pt_qw_avg.item()), it)
        writer.add_scalar("train/loss_pt_coord_weighted", float(loss_pt_coordw_avg.item()), it)
        writer.add_scalar("train/loss_pt_link_weighted", float(loss_pt_linkw_avg.item()), it)
        writer.add_scalar("train/lr", float(lr), it)

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

        if args.dataset_type == "voc" and args.viz_every > 0 and (it % args.viz_every == 0):
            _save_train_viz(
                out_dir=viz_dir,
                it=it,
                images=images,
                targets_list=targets_list,
                outs=outs,
                loss_value=float(loss.item()),
                loss_breakdown=loss_breakdown,
                suggested_weights=suggested_weights,
                viz_topk=args.viz_topk,
                viz_topk_after_nms=args.viz_topk_after_nms,
                conf_thres=args.conf_thres,
                nms_iou_threshold=args.nms_iou_threshold,
                nms_max_dets=args.nms_max_dets,
                image_size=args.image_size,
            )
            writer.add_text("train/viz", f"Saved train visualization at iter={it}", it)

        if it % args.log_every == 0:
            avg = running / float(args.log_every)
            running = 0.0
            logger.info("Iter %d/%d | lr=%.6f | loss=%.6f", it, total_iters, lr, avg)

        if args.save_every > 0 and (it % args.save_every == 0):
            step_ckpt = model_dir / f"pln_step_{it:06d}.pth"
            torch.save(
                {
                    "iter": it,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "timestamp": timestamp,
                },
                step_ckpt,
            )
            latest_ckpt = model_dir / "pln_latest.pth"
            torch.save(
                {
                    "iter": it,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "timestamp": timestamp,
                },
                latest_ckpt,
            )
            logger.info("Saved checkpoint: %s", step_ckpt)

    final_ckpt = model_dir / f"pln_{args.dataset_type}_{args.split}_final.pth"
    torch.save(
        {
            "iter": it,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "timestamp": timestamp,
        },
        final_ckpt,
    )
    latest_ckpt = model_dir / "pln_latest.pth"
    torch.save(
        {
            "iter": it,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "timestamp": timestamp,
        },
        latest_ckpt,
    )
    pbar.close()
    writer.close()
    logger.info("Saved final checkpoint: %s", final_ckpt)
    logger.info("Run complete.")


if __name__ == "__main__":
    main()

