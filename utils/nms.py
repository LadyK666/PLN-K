from __future__ import annotations

from typing import Optional

import torch


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes in xyxy format.

    Args:
        boxes1: (M,4)
        boxes2: (N,4)
    Returns:
        iou: (M,N)
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=boxes1.dtype)

    x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + eps)


def nms_single_class(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    *,
    max_dets: Optional[int] = None,
) -> torch.Tensor:
    """
    Greedy NMS for one class.

    Args:
        boxes: (N,4)
        scores: (N,)
    Returns:
        keep_indices: 1D tensor of kept indices (in the original indexing)
    """
    if boxes.numel() == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.long)

    # Sort by score desc
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        ious = box_iou_xyxy(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[ious <= iou_threshold]

        if max_dets is not None and len(keep) >= max_dets:
            break

    return torch.stack(keep) if len(keep) > 0 else torch.empty((0,), device=boxes.device, dtype=torch.long)


def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
    *,
    max_dets: Optional[int] = None,
) -> torch.Tensor:
    """
    Class-aware NMS: run NMS separately per label, then merge kept indices.
    Optionally apply global max_dets by score after merging.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.long)

    labels = labels.to(device=boxes.device)
    kept_idx = []
    unique_labels = torch.unique(labels)
    for lab in unique_labels.tolist():
        idx = torch.nonzero(labels == lab, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        keep_local = nms_single_class(
            boxes[idx],
            scores[idx],
            iou_threshold,
            max_dets=max_dets,
        )
        kept_idx.append(idx[keep_local])

    if len(kept_idx) == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.long)

    kept_idx = torch.cat(kept_idx, dim=0)

    # Global max_dets by score
    if max_dets is not None and kept_idx.numel() > max_dets:
        kept_scores = scores[kept_idx]
        topk = torch.topk(kept_scores, k=max_dets).indices
        kept_idx = kept_idx[topk]

    return kept_idx

