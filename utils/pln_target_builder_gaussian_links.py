from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

import torch


BranchName = Literal["left_top", "right_top", "left_bottom", "right_bottom"]

VOC_CLASSES: List[str] = [
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


# Priority order used when a grid cell has more than 2 points.
# This is taken from `debug_out/grid_point_stats_report.md` ranking (73-92).
CLASS_PRIORITY: List[str] = [
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


@dataclass(frozen=True)
class _Point:
    kind: Literal["center", "corner"]
    cls: str
    area: float
    x_r: float  # resized image coordinate [0,image_size]
    y_r: float
    u: int
    v: int
    dx: float  # in-cell coord [0,1]
    dy: float
    # 链路的「目标格」索引 (u_t, v_t)，用于构造 GT 的 Lx/Ly 因子化 one-hot/gaussian
    link_u: int  # 目标格在水平方向的格索引 u_t ∈ [0, S-1]，对应张量 Lx 的第 k 维
    link_v: int  # 目标格在竖直方向的格索引 v_t ∈ [0, S-1]，对应张量 Ly 的第 k 维


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
) -> Tuple[float, float, int, int, float, float]:
    """
    Map original image coordinate (x,y) into resized coordinate (x_r,y_r),
    then to grid cell (u,v) and in-cell normalized offsets (dx,dy) in [0,1].

    Offset convention: relative to the cell right-top corner.
      x_rt = (u+1)*stride, y_rt = v*stride
      dx = (x_rt - x_r)/stride
      dy = (y_r - y_rt)/stride
    """
    x_r = x / src_w * image_size
    y_r = y / src_h * image_size
    stride = image_size / grid_size

    u = int(x_r // stride)
    v = int(y_r // stride)
    u = _clamp_int(u, 0, grid_size - 1)
    v = _clamp_int(v, 0, grid_size - 1)

    x_rt = (u + 1) * stride
    y_rt = v * stride
    dx = (x_rt - x_r) / stride
    dy = (y_r - y_rt) / stride
    dx = float(max(0.0, min(1.0, dx)))
    dy = float(max(0.0, min(1.0, dy)))

    return x_r, y_r, u, v, dx, dy


def _pick_branch_corner_xy(
    x1: float, y1: float, x2: float, y2: float, branch: BranchName
) -> Tuple[float, float]:
    if branch == "left_top":
        return x1, y1
    if branch == "right_top":
        return x2, y1
    if branch == "left_bottom":
        return x1, y2
    if branch == "right_bottom":
        return x2, y2
    raise ValueError(f"Unknown branch={branch}")


def _point_sort_key(p: _Point) -> Tuple[int, float]:
    # smaller rank is higher priority; for same class: smaller box area first
    return (_PRIORITY_RANK.get(p.cls, 10**9), float(p.area))


def _select_top2(points: List[_Point]) -> List[_Point]:
    if len(points) <= 2:
        return points
    points_sorted = sorted(points, key=_point_sort_key)
    return points_sorted[:2]


def _gaussian_1d(dist: int, sigma: float) -> float:
    """
    1D Gaussian weight by Euclidean distance on the 1D k-axis:
      w = exp( -d^2 / (2*sigma^2) )
    dist is integer (>=0).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    d2 = float(dist * dist)
    return math.exp(-d2 / (2.0 * float(sigma) * float(sigma)))


def _write_gaussian_link_1d(
    *,
    L_hat: torch.Tensor,  # (P, S, S, S)
    p_slot: int,
    k0: int,
    v: int,
    u: int,
    S: int,
    gaussian_radius: int,
    gaussian_sigma: float,
) -> None:
    """
    Write 1D gaussian targets on k dimension for one (p_slot, v_src=u?, u_src) location.

    NOTE:
    - L_hat layout: [p, k, v_src, u_src]
    - We only affect k dimension around k0.
    """
    if gaussian_radius < 0:
        raise ValueError(f"gaussian_radius must be >=0, got {gaussian_radius}")
    k_lo = max(0, k0 - gaussian_radius)
    k_hi = min(S - 1, k0 + gaussian_radius)
    for k in range(k_lo, k_hi + 1):
        dist = abs(k - k0)
        L_hat[p_slot, k, v, u] = float(_gaussian_1d(dist=dist, sigma=gaussian_sigma))


def build_pln_targets_for_branch_from_resized_boxes_gaussian_links(
    *,
    boxes_xyxy_resized: torch.Tensor,
    labels_idx: torch.Tensor,
    branch: BranchName,
    image_size: int = 448,
    grid_size: int = 14,
    B_point: int = 2,
    gaussian_radius: int = 1,
    gaussian_sigma: float = 1.0,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Build PLN targets for ONE image and ONE branch, but make Lx/Ly link supervision
    a gaussian neighborhood around the true target (link_u/link_v) instead of
    hard one-hot.

    Lx/Ly are still factorized:
      - Lx_hat[p, k, v_src, u_src] is a gaussian over k around link_u
      - Ly_hat[p, k, v_src, u_src] is a gaussian over k around link_v
    """
    if device is None:
        device = (
            boxes_xyxy_resized.device
            if isinstance(boxes_xyxy_resized, torch.Tensor)
            else torch.device("cpu")
        )

    S = grid_size
    N = len(VOC_CLASSES)
    P = 2 * B_point
    if P != 4:
        raise ValueError("This implementation assumes B_point=2 -> P=4 slots.")

    boxes = boxes_xyxy_resized.to(device=device, dtype=torch.float32)
    labels_idx = labels_idx.to(device=device, dtype=torch.int64)

    stride = float(image_size) / float(S)

    centers_by_cell: Dict[Tuple[int, int], List[_Point]] = {}
    corners_by_cell: Dict[Tuple[int, int], List[_Point]] = {}

    for b in range(int(boxes.shape[0])):
        x1, y1, x2, y2 = [float(v) for v in boxes[b].tolist()]
        cls_i = int(labels_idx[b].item())
        if cls_i < 0 or cls_i >= len(VOC_CLASSES):
            continue
        cls = VOC_CLASSES[cls_i]
        area = float(max(0.0, (x2 - x1) * (y2 - y1)))

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        u_c = _clamp_int(int(cx // stride), 0, S - 1)
        v_c = _clamp_int(int(cy // stride), 0, S - 1)
        x_rt_c = (u_c + 1) * stride
        y_rt_c = v_c * stride
        dx_c = float((x_rt_c - cx) / stride)
        dy_c = float((cy - y_rt_c) / stride)
        dx_c = float(max(0.0, min(1.0, dx_c)))
        dy_c = float(max(0.0, min(1.0, dy_c)))

        # branch-specific corner point on resized grid
        kx, ky = _pick_branch_corner_xy(x1, y1, x2, y2, branch)
        u_k = _clamp_int(int(kx // stride), 0, S - 1)
        v_k = _clamp_int(int(ky // stride), 0, S - 1)
        x_rt_k = (u_k + 1) * stride
        y_rt_k = v_k * stride
        dx_k = float((x_rt_k - kx) / stride)
        dy_k = float((ky - y_rt_k) / stride)
        dx_k = float(max(0.0, min(1.0, dx_k)))
        dy_k = float(max(0.0, min(1.0, dy_k)))

        centers_by_cell.setdefault((u_c, v_c), []).append(
            _Point(
                kind="center",
                cls=cls,
                area=area,
                x_r=cx,
                y_r=cy,
                u=u_c,
                v=v_c,
                dx=dx_c,
                dy=dy_c,
                link_u=u_k,
                link_v=v_k,
            )
        )

        corners_by_cell.setdefault((u_k, v_k), []).append(
            _Point(
                kind="corner",
                cls=cls,
                area=area,
                x_r=kx,
                y_r=ky,
                u=u_k,
                v=v_k,
                dx=dx_k,
                dy=dy_k,
                link_u=u_c,
                link_v=v_c,
            )
        )

    pt_mask = torch.zeros((P, S, S), dtype=torch.float32, device=device)
    nopt_mask = torch.ones((P, S, S), dtype=torch.float32, device=device)

    Q_hat = torch.zeros((P, N, S, S), dtype=torch.float32, device=device)
    x_hat = torch.zeros((P, S, S), dtype=torch.float32, device=device)
    y_hat = torch.zeros((P, S, S), dtype=torch.float32, device=device)

    # [p, k, v_src, u_src]
    Lx_hat = torch.zeros((P, S, S, S), dtype=torch.float32, device=device)
    Ly_hat = torch.zeros((P, S, S, S), dtype=torch.float32, device=device)

    for (u, v), pts in centers_by_cell.items():
        chosen = _select_top2(pts)
        for slot_idx, pnt in enumerate(chosen):
            p_slot = slot_idx  # 0 or 1
            pt_mask[p_slot, v, u] = 1.0
            nopt_mask[p_slot, v, u] = 0.0
            Q_hat[p_slot, _CLASS_TO_IDX[pnt.cls], v, u] = 1.0
            x_hat[p_slot, v, u] = float(pnt.dx)
            y_hat[p_slot, v, u] = float(pnt.dy)

            # Replace hard one-hot at link_u/link_v with gaussian neighborhood.
            _write_gaussian_link_1d(
                L_hat=Lx_hat,
                p_slot=p_slot,
                k0=pnt.link_u,
                v=v,
                u=u,
                S=S,
                gaussian_radius=gaussian_radius,
                gaussian_sigma=gaussian_sigma,
            )
            _write_gaussian_link_1d(
                L_hat=Ly_hat,
                p_slot=p_slot,
                k0=pnt.link_v,
                v=v,
                u=u,
                S=S,
                gaussian_radius=gaussian_radius,
                gaussian_sigma=gaussian_sigma,
            )

    for (u, v), pts in corners_by_cell.items():
        chosen = _select_top2(pts)
        for slot_idx, pnt in enumerate(chosen):
            p_slot = B_point + slot_idx  # 2 or 3
            pt_mask[p_slot, v, u] = 1.0
            nopt_mask[p_slot, v, u] = 0.0
            Q_hat[p_slot, _CLASS_TO_IDX[pnt.cls], v, u] = 1.0
            x_hat[p_slot, v, u] = float(pnt.dx)
            y_hat[p_slot, v, u] = float(pnt.dy)

            _write_gaussian_link_1d(
                L_hat=Lx_hat,
                p_slot=p_slot,
                k0=pnt.link_u,
                v=v,
                u=u,
                S=S,
                gaussian_radius=gaussian_radius,
                gaussian_sigma=gaussian_sigma,
            )
            _write_gaussian_link_1d(
                L_hat=Ly_hat,
                p_slot=p_slot,
                k0=pnt.link_v,
                v=v,
                u=u,
                S=S,
                gaussian_radius=gaussian_radius,
                gaussian_sigma=gaussian_sigma,
            )

    P_hat = pt_mask.clone()
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

