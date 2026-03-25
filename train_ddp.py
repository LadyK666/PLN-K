from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
from train import _save_train_viz  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("PLN Train DDP")
    p.add_argument("--dataset_type", type=str, default="voc", choices=["voc", "image_only"])
    p.add_argument("--data_root", type=str, default="")
    default_voc2007_root = str(ROOT / "VOC2007" / "VOCdevkit" / "VOC2007")
    p.add_argument("--voc2007_root", type=str, default=default_voc2007_root)
    default_voc2012_root = str(ROOT / "VOC2012" / "VOCdevkit" / "VOC2012")
    p.add_argument("--voc2012_root", type=str, default=default_voc2012_root)
    p.add_argument("--split", type=str, default="trainval")
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=64, help="Global batch size across all GPUs")
    p.add_argument("--lr_start", type=float, default=0.001)
    p.add_argument("--lr_end", type=float, default=0.005)
    p.add_argument("--warmup_iters", type=int, default=10000)
    p.add_argument("--finetune_iters", type=int, default=30000)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.00004)
    p.add_argument("--pretrained_backbone", action="store_true")
    p.add_argument(
        "--train_backbone",
        action="store_true",
        help="If set, backbone parameters are trainable (can still be initialized by --pretrained_backbone).",
    )
    p.add_argument("--no_freeze_bn", action="store_true")
    p.add_argument("--no_voc2012_train", action="store_true")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--viz_every", type=int, default=10)
    p.add_argument("--viz_topk", type=int, default=60)
    p.add_argument("--viz_topk_after_nms", type=int, default=60)
    p.add_argument("--conf_thres", type=float, default=0.25)
    p.add_argument("--nms_iou_threshold", type=float, default=0.45)
    p.add_argument("--nms_max_dets", type=int, default=60)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--save_latest_every", type=int, default=50, help="Save rolling latest checkpoint every N iterations")
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    p.add_argument("--loss_logit_clip", type=float, default=20.0)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--resume", type=str, default="", help="Resume full training state from checkpoint")
    p.add_argument("--loss_input", type=str, default="logits", choices=["logits", "normalized"])
    p.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--master_port", type=str, default="29501")

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


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _is_main() -> bool:
    return _rank() == 0


def _setup_dist(args: argparse.Namespace) -> int:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_PORT", args.master_port)
        dist.init_process_group(backend=args.backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def _cleanup_dist() -> None:
    if _is_dist():
        dist.destroy_process_group()


def _reduce_mean_scalar(x: float, device: torch.device) -> float:
    t = torch.tensor([x], dtype=torch.float32, device=device)
    if _is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= float(_world())
    return float(t.item())


def main() -> None:
    args = parse_args()
    local_rank = _setup_dist(args)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    transform = Compose(
        [
            Resize((args.image_size, args.image_size), antialias=True),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if args.dataset_type == "voc":
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
        collate_fn = collate_voc_detection
    else:
        ds = ImageOnlyDataset(root_dir=args.data_root, split_dir=args.split, transform=transform)
        collate_fn = collate_images

    sampler = DistributedSampler(ds, shuffle=True) if _world() > 1 else None
    per_rank_batch = max(1, args.batch_size // _world())
    dl = DataLoader(
        ds,
        batch_size=per_rank_batch,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    model = PLNModel(
        backbone_pretrained=args.pretrained_backbone,
        backbone_trainable=args.train_backbone,
        freeze_bn=not args.no_freeze_bn,
    ).to(device)

    resume_state = None
    start_iter = 0
    if args.resume:
        resume_state = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_state.get("model", resume_state), strict=False)
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    if not args.train_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    if _world() > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adam":
        optimizer = Adam(trainable_params, lr=args.lr_start, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(trainable_params, lr=args.lr_start, momentum=args.momentum, weight_decay=args.weight_decay)

    if resume_state is not None and "optimizer" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        start_iter = int(resume_state.get("iter", 0))

    criterion = PLNLoss(
        weights=PLNLossWeights(w_class=1.0, w_coord=1.0, w_link=1.0),
        reduction="mean",
        logit_clip=(args.loss_logit_clip if args.loss_logit_clip > 0 else None),
    )

    total_iters = args.warmup_iters + args.finetune_iters
    it = start_iter
    running = 0.0

    base_out_dir = Path(args.voc2007_root if args.dataset_type == "voc" else args.data_root).parent / "pln_runs"
    if _is_main():
        base_out_dir.mkdir(parents=True, exist_ok=True)
        if args.resume:
            resume_p = Path(args.resume).resolve()
            # expected: .../run_xxx/models/pln_*.pth
            run_dir = resume_p.parent.parent
            if not run_dir.exists():
                raise FileNotFoundError(f"Cannot infer run_dir from resume: {args.resume}")
            timestamp = run_dir.name.replace("run_", "")
        else:
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
        config_payload = vars(args).copy()
        config_payload["start_iter"] = start_iter
        with open(run_dir / "config.json", "w") as f:
            json.dump(config_payload, f, indent=2)
        logger = logging.getLogger("pln_train_ddp")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(run_dir / "train.log")
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
        logger.info("DDP world_size=%d local_batch=%d global_batch=%d", _world(), per_rank_batch, per_rank_batch * _world())
        logger.info("Resume mode: %s | start_iter=%d", bool(args.resume), start_iter)
        logger.info("Run directory: %s", run_dir)
    else:
        run_dir = model_dir = viz_dir = tb_dir = None
        writer = None
        logger = None

    model.train()
    branch_keys = ["left_top", "right_top", "left_bottom", "right_bottom"]
    dl_iter = iter(dl)
    pbar = tqdm(total=total_iters, initial=it, desc="PLN Train DDP", dynamic_ncols=True, disable=(not _is_main()))

    while it < total_iters:
        if sampler is not None and (it % max(1, len(dl)) == 0):
            sampler.set_epoch(it // max(1, len(dl)))
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        images = (batch["images"] if args.dataset_type == "voc" else batch["image"]).to(device)
        targets_list = batch["targets"] if args.dataset_type == "voc" else None

        lr = args.lr_start + (args.lr_end - args.lr_start) * (it / float(max(1, args.warmup_iters))) if it < args.warmup_iters else args.lr_end
        for group in optimizer.param_groups:
            group["lr"] = lr

        outs = model(images)
        if args.dataset_type != "voc":
            raise RuntimeError("train_ddp.py currently supports loss only for VOC dataset_type.")

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

        for branch in branch_keys:
            pred_map = outs[branch]
            if args.loss_input == "normalized":
                pred_dec = decode_branch_channels_inference(pred_map, s=14, num_classes=20, num_points=4)
                pred_for_loss = {"P": pred_dec["P"], "Q": pred_dec["Q"], "x": pred_dec["dx"], "y": pred_dec["dy"], "Lx": pred_dec["Lx"], "Ly": pred_dec["Ly"]}
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

            B = images.shape[0]
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
                if args.use_gaussian_link_targets:
                    built = build_pln_targets_for_branch_from_resized_boxes_gaussian_links(
                        boxes_xyxy_resized=t["boxes"].to(device=device, dtype=torch.float32),
                        labels_idx=t["labels"].to(device=device, dtype=torch.int64),
                        branch=branch,  # type: ignore[arg-type]
                        image_size=args.image_size,
                        grid_size=14,
                        B_point=2,
                        gaussian_radius=args.gaussian_link_radius,
                        gaussian_sigma=args.gaussian_link_sigma,
                        device=device,
                    )
                else:
                    built = build_pln_targets_for_branch_from_resized_boxes(
                        boxes_xyxy_resized=t["boxes"].to(device=device, dtype=torch.float32),
                        labels_idx=t["labels"].to(device=device, dtype=torch.int64),
                        branch=branch,  # type: ignore[arg-type]
                        image_size=args.image_size,
                        grid_size=14,
                        B_point=2,
                        device=device,
                    )
                P_hat[bi], Q_hat[bi], x_hat[bi], y_hat[bi] = built["P"], built["Q"], built["x"], built["y"]
                Lx_hat[bi], Ly_hat[bi] = built["Lx"], built["Ly"]
                pt_mask[bi], nopt_mask[bi] = built["pt_mask"], built["nopt_mask"]

            loss_dict = criterion(
                pred=pred_for_loss,
                target={"P": P_hat, "Q": Q_hat, "x": x_hat, "y": y_hat, "Lx": Lx_hat, "Ly": Ly_hat},
                pt_mask=pt_mask,
                nopt_mask=nopt_mask,
            )
            loss_total += loss_dict["loss"]
            loss_pt_total += loss_dict["loss_pt"]
            loss_nopt_total += loss_dict["loss_nopt"]
            loss_pt_p_total += loss_dict["loss_pt_p"]
            loss_pt_qw_total += loss_dict["loss_pt_q_weighted"]
            loss_pt_coordw_total += loss_dict["loss_pt_coord_weighted"]
            loss_pt_linkw_total += loss_dict["loss_pt_link_weighted"]
            loss_pt_qraw_total += loss_dict["loss_pt_q_raw"]
            loss_pt_coordraw_total += loss_dict["loss_pt_coord_raw"]
            loss_pt_linkraw_total += loss_dict["loss_pt_link_raw"]

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

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
        optimizer.step()

        # IMPORTANT: all distributed reductions must be called on every rank.
        loss_item = _reduce_mean_scalar(float(loss.item()), device)
        loss_pt_item = _reduce_mean_scalar(float(loss_pt_avg.item()), device)
        loss_nopt_item = _reduce_mean_scalar(float(loss_nopt_avg.item()), device)
        loss_pt_p_item = _reduce_mean_scalar(float(loss_pt_p_avg.item()), device)
        loss_pt_qw_item = _reduce_mean_scalar(float(loss_pt_qw_avg.item()), device)
        loss_pt_coordw_item = _reduce_mean_scalar(float(loss_pt_coordw_avg.item()), device)
        loss_pt_linkw_item = _reduce_mean_scalar(float(loss_pt_linkw_avg.item()), device)
        loss_pt_qraw_item = _reduce_mean_scalar(float(loss_pt_qraw_avg.item()), device)
        loss_pt_coordraw_item = _reduce_mean_scalar(float(loss_pt_coordraw_avg.item()), device)
        loss_pt_linkraw_item = _reduce_mean_scalar(float(loss_pt_linkraw_avg.item()), device)

        running += loss_item
        it += 1
        if _is_main():
            pbar.update(1)
            pbar.set_postfix({"lr": f"{lr:.6f}", "loss": f"{loss_item:.4f}"})
            writer.add_scalar("train/loss", loss_item, it)
            writer.add_scalar("train/loss_pt", loss_pt_item, it)
            writer.add_scalar("train/loss_nopt", loss_nopt_item, it)
            writer.add_scalar("train/loss_pt_p", loss_pt_p_item, it)
            writer.add_scalar("train/loss_pt_q_weighted", loss_pt_qw_item, it)
            writer.add_scalar("train/loss_pt_coord_weighted", loss_pt_coordw_item, it)
            writer.add_scalar("train/loss_pt_link_weighted", loss_pt_linkw_item, it)
            writer.add_scalar("train/lr", float(lr), it)

            total_now = loss_item + 1e-8
            loss_breakdown = {
                "p": loss_pt_p_item / total_now,
                "q": loss_pt_qw_item / total_now,
                "coord": loss_pt_coordw_item / total_now,
                "link": loss_pt_linkw_item / total_now,
                "nopt": loss_nopt_item / total_now,
            }
            p_base = max(loss_pt_p_item, 1e-8)
            suggested_weights = {
                "w_class": p_base / max(loss_pt_qraw_item, 1e-8),
                "w_coord": p_base / max(loss_pt_coordraw_item, 1e-8),
                "w_link": p_base / max(loss_pt_linkraw_item, 1e-8),
            }

            if args.viz_every > 0 and (it % args.viz_every == 0):
                _save_train_viz(
                    out_dir=viz_dir,
                    it=it,
                    images=images,
                    targets_list=targets_list,
                    outs=outs,
                    loss_value=loss_item,
                    loss_breakdown=loss_breakdown,
                    suggested_weights=suggested_weights,
                    viz_topk=args.viz_topk,
                    viz_topk_after_nms=args.viz_topk_after_nms,
                    conf_thres=args.conf_thres,
                    nms_iou_threshold=args.nms_iou_threshold,
                    nms_max_dets=args.nms_max_dets,
                    image_size=args.image_size,
                )

            if it % args.log_every == 0:
                avg = running / float(args.log_every)
                running = 0.0
                logger.info("Iter %d/%d | lr=%.6f | loss=%.6f", it, total_iters, lr, avg)

            if args.save_every > 0 and (it % args.save_every == 0):
                ckpt = {
                    "iter": it,
                    "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "run_dir": str(run_dir),
                }
                torch.save(ckpt, model_dir / f"pln_step_{it:06d}.pth")
                torch.save(ckpt, model_dir / "pln_latest.pth")
            elif args.save_latest_every > 0 and (it % args.save_latest_every == 0):
                ckpt = {
                    "iter": it,
                    "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "run_dir": str(run_dir),
                }
                torch.save(ckpt, model_dir / "pln_latest.pth")

    if _is_main():
        final_ckpt = {
            "iter": it,
            "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "run_dir": str(run_dir),
        }
        torch.save(final_ckpt, model_dir / f"pln_{args.dataset_type}_{args.split}_final.pth")
        torch.save(final_ckpt, model_dir / "pln_latest.pth")
        pbar.close()
        writer.close()
        logger.info("Run complete.")

    _cleanup_dist()


if __name__ == "__main__":
    main()

