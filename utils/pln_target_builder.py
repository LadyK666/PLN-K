from __future__ import annotations

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
    # 链路的「目标格」索引 (u_t, v_t)，用于构造 GT 的 Lx/Ly 因子化 one-hot（见下方写入逻辑）
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
    # numeric safety
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


def build_pln_targets_for_branch(
    *,
    boxes_xyxy: Sequence[Sequence[float]],
    labels: Sequence[str],
    image_width: float,
    image_height: float,
    branch: BranchName,
    image_size: int = 448,
    grid_size: int = 14,
    B_point: int = 2,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Build PLN targets for ONE image and ONE branch.

    Output tensor shapes match `PLNLoss` expectation (without batch dimension):
      - P/Q/x/y: (P=4, S, S) or (P=4, N, S, S)
      - Lx/Ly:   (P=4, S, S, S) with layout [p, k, v, u]
      - pt_mask / nopt_mask: (P=4, S, S)

    Slot convention (P=4):
      - p=0,1: center slots (j=1,2)
      - p=2,3: corner slots (j=3,4) AFTER branch-specific corner filtering
    """
    if device is None:
        device = torch.device("cpu")

    S = grid_size
    N = len(VOC_CLASSES)
    P = 2 * B_point  # should be 4 when B_point=2
    if P != 4:
        raise ValueError("This implementation assumes B_point=2 -> P=4 slots.")

    # Accumulate candidate points per cell for centers and corners separately.
    centers_by_cell: Dict[Tuple[int, int], List[_Point]] = {}
    corners_by_cell: Dict[Tuple[int, int], List[_Point]] = {}

    for (x1, y1, x2, y2), cls in zip(boxes_xyxy, labels):
        if cls not in _CLASS_TO_IDX:
            continue
        area = float(max(0.0, (x2 - x1) * (y2 - y1)))

        # center
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        _, _, u_c, v_c, dx_c, dy_c = _to_grid_and_incell(
            cx,
            cy,
            src_w=image_width,
            src_h=image_height,
            image_size=image_size,
            grid_size=S,
        )

        # 中心点的 Lx/Ly 监督：链到「同一框、当前 branch 选定的那一个角点」所在的格 (u_k, v_k)
        kx, ky = _pick_branch_corner_xy(x1, y1, x2, y2, branch)  # 在原始图上取该分支对应的角点像素坐标
        _, _, u_k, v_k, _, _ = _to_grid_and_incell(  # 把角点映射到 resize 图上的格索引与格内偏移（此处只用 u_k,v_k）
            kx,
            ky,
            src_w=image_width,
            src_h=image_height,
            image_size=image_size,
            grid_size=S,
        )

        p_center = _Point(
            kind="center",
            cls=cls,
            area=area,
            x_r=cx / image_width * image_size,  # 中心在 resize 图上的 x（像素）
            y_r=cy / image_height * image_size,  # 中心在 resize 图上的 y（像素）
            u=u_c,  # 中心所在格 u
            v=v_c,  # 中心所在格 v
            dx=dx_c,  # 中心格内归一化 dx
            dy=dy_c,  # 中心格内归一化 dy
            link_u=u_k,  # GT：Lx 在 k=u_k 处为 1，表示水平方向链到角点列 u_k
            link_v=v_k,  # GT：Ly 在 k=v_k 处为 1，表示竖直方向链到角点行 v_k
        )
        centers_by_cell.setdefault((u_c, v_c), []).append(p_center)

        # corners: only keep the branch-specific corner type (left_top etc.)
        x_corner, y_corner = kx, ky
        _, _, u_corner, v_corner, dx_corner, dy_corner = _to_grid_and_incell(
            x_corner,
            y_corner,
            src_w=image_width,
            src_h=image_height,
            image_size=image_size,
            grid_size=S,
        )
        # 角点的 Lx/Ly 监督：链回「同一框中心点」所在的格 (u_c, v_c)，与中心→角点互为配对
        p_corner = _Point(
            kind="corner",
            cls=cls,
            area=area,
            x_r=x_corner / image_width * image_size,  # 角点在 resize 图上的 x
            y_r=y_corner / image_height * image_size,  # 角点在 resize 图上的 y
            u=u_corner,  # 角点所在格 u（即 u_k）
            v=v_corner,  # 角点所在格 v（即 v_k）
            dx=dx_corner,  # 角点格内 dx
            dy=dy_corner,  # 角点格内 dy
            link_u=u_c,  # GT：Lx 在 k=u_c 处为 1，角点水平方向链到中心列 u_c
            link_v=v_c,  # GT：Ly 在 k=v_c 处为 1，角点竖直方向链到中心行 v_c
        )
        corners_by_cell.setdefault((u_corner, v_corner), []).append(p_corner)

    # Allocate outputs
    pt_mask = torch.zeros((P, S, S), dtype=torch.float32, device=device)
    nopt_mask = torch.ones((P, S, S), dtype=torch.float32, device=device)

    Q_hat = torch.zeros((P, N, S, S), dtype=torch.float32, device=device)
    x_hat = torch.zeros((P, S, S), dtype=torch.float32, device=device)
    y_hat = torch.zeros((P, S, S), dtype=torch.float32, device=device)
    # Lx_hat/Ly_hat：形状 [P, S, S, S]，布局 [p, k, v_src, u_src]；在源格 (u_src,v_src) 上对「目标格」做因子化 one-hot
    # - Lx[p, k, v_src, u_src]=1 表示 k 为目标格的水平索引 u_t（与 decoder 中 Lx 最后一维含义一致）
    # - Ly[p, k, v_src, u_src]=1 表示 k 为目标格的竖直索引 v_t；联合即完整目标格 (u_t, v_t)
    Lx_hat = torch.zeros((P, S, S, S), dtype=torch.float32, device=device)  # [p, k, v, u]
    Ly_hat = torch.zeros((P, S, S, S), dtype=torch.float32, device=device)  # [p, k, v, u]

    # Fill center slots p=0,1 per cell
    for (u, v), pts in centers_by_cell.items():
        chosen = _select_top2(pts)
        for slot_idx, p in enumerate(chosen):
            p_slot = slot_idx  # 0 or 1
            pt_mask[p_slot, v, u] = 1.0
            nopt_mask[p_slot, v, u] = 0.0

            Q_hat[p_slot, _CLASS_TO_IDX[p.cls], v, u] = 1.0
            x_hat[p_slot, v, u] = float(p.dx)
            y_hat[p_slot, v, u] = float(p.dy)

            # 在源格 (u,v) 上写入链路 GT：Lx/Ly 各在 k 维上单独 one-hot，乘积可还原目标格 (link_u, link_v)
            Lx_hat[p_slot, p.link_u, v, u] = 1.0  # 第 k=link_u 个水平槽置 1
            Ly_hat[p_slot, p.link_v, v, u] = 1.0  # 第 k=link_v 个竖直槽置 1

    # Fill corner slots p=2,3 per cell (after branch corner filtering already applied)
    for (u, v), pts in corners_by_cell.items():
        chosen = _select_top2(pts)
        for slot_idx, p in enumerate(chosen):
            p_slot = B_point + slot_idx  # 2 or 3
            pt_mask[p_slot, v, u] = 1.0
            nopt_mask[p_slot, v, u] = 0.0

            Q_hat[p_slot, _CLASS_TO_IDX[p.cls], v, u] = 1.0
            x_hat[p_slot, v, u] = float(p.dx)
            y_hat[p_slot, v, u] = float(p.dy)

            # 角点槽：同样在 (u,v) 上写因子化 one-hot，目标为中心格 (link_u, link_v)
            Lx_hat[p_slot, p.link_u, v, u] = 1.0
            Ly_hat[p_slot, p.link_v, v, u] = 1.0

    # P_hat for convenience: 1 for pt, 0 for nopt
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


def build_pln_targets_for_branch_from_resized_boxes(
    *,
    boxes_xyxy_resized: torch.Tensor,
    labels_idx: torch.Tensor,
    branch: BranchName,
    image_size: int = 448,
    grid_size: int = 14,
    B_point: int = 2,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Build PLN targets for ONE image and ONE branch, assuming boxes are already
    in the resized pixel coordinate system of size (image_size, image_size).

    This matches `VOC2007Dataset` behavior when output_size=(image_size,image_size).
    """
    if device is None:
        device = boxes_xyxy_resized.device if isinstance(boxes_xyxy_resized, torch.Tensor) else torch.device("cpu")

    S = grid_size
    N = len(VOC_CLASSES)
    P = 2 * B_point
    if P != 4:
        raise ValueError("This implementation assumes B_point=2 -> P=4 slots.")

    boxes = boxes_xyxy_resized.to(device=device, dtype=torch.float32)
    labels_idx = labels_idx.to(device=device, dtype=torch.int64)

    # Python-side grouping is acceptable for VOC scale and keeps logic identical
    # to the specification (priority + area sorting).
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

        # 当前分支对应的框角点 (kx,ky)，用于中心→角点的 Lx/Ly 目标格 (u_k,v_k)
        kx, ky = _pick_branch_corner_xy(x1, y1, x2, y2, branch)
        u_k = _clamp_int(int(kx // stride), 0, S - 1)  # 角点落在第几列格（水平索引）
        v_k = _clamp_int(int(ky // stride), 0, S - 1)  # 角点落在第几行格（竖直索引）
        x_rt_k = (u_k + 1) * stride  # 角点所在格右上角 x（与 _to_grid_and_incell 约定一致）
        y_rt_k = v_k * stride  # 角点所在格右上角 y
        dx_k = float((x_rt_k - kx) / stride)  # 角点格内 dx
        dy_k = float((ky - y_rt_k) / stride)  # 角点格内 dy
        dx_k = float(max(0.0, min(1.0, dx_k)))  # 夹紧到 [0,1]
        dy_k = float(max(0.0, min(1.0, dy_k)))

        # 中心槽：源格 (u_c,v_c)，链路目标格为角点格 (u_k,v_k) → 写入时 Lx[:,u_k], Ly[:,v_k]
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

        # 角点槽：源格 (u_k,v_k)，链路目标格为中心格 (u_c,v_c)
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
    # 与 build_pln_targets_for_branch 相同：Lx/Ly 为 [p,k,v_src,u_src]，k 维因子化表示目标格
    Lx_hat = torch.zeros((P, S, S, S), dtype=torch.float32, device=device)  # [p, k, v, u]
    Ly_hat = torch.zeros((P, S, S, S), dtype=torch.float32, device=device)  # [p, k, v, u]

    for (u, v), pts in centers_by_cell.items():
        chosen = _select_top2(pts)
        for slot_idx, pnt in enumerate(chosen):
            p_slot = slot_idx  # 0 or 1
            pt_mask[p_slot, v, u] = 1.0
            nopt_mask[p_slot, v, u] = 0.0
            Q_hat[p_slot, _CLASS_TO_IDX[pnt.cls], v, u] = 1.0
            x_hat[p_slot, v, u] = float(pnt.dx)
            y_hat[p_slot, v, u] = float(pnt.dy)
            Lx_hat[p_slot, pnt.link_u, v, u] = 1.0  # 中心点：水平 one-hot 指向角点列 pnt.link_u
            Ly_hat[p_slot, pnt.link_v, v, u] = 1.0  # 中心点：竖直 one-hot 指向角点行 pnt.link_v

    for (u, v), pts in corners_by_cell.items():
        chosen = _select_top2(pts)
        for slot_idx, pnt in enumerate(chosen):
            p_slot = B_point + slot_idx  # 2 or 3
            pt_mask[p_slot, v, u] = 1.0
            nopt_mask[p_slot, v, u] = 0.0
            Q_hat[p_slot, _CLASS_TO_IDX[pnt.cls], v, u] = 1.0
            x_hat[p_slot, v, u] = float(pnt.dx)
            y_hat[p_slot, v, u] = float(pnt.dy)
            Lx_hat[p_slot, pnt.link_u, v, u] = 1.0  # 角点：水平 one-hot 指向中心列 pnt.link_u
            Ly_hat[p_slot, pnt.link_v, v, u] = 1.0  # 角点：竖直 one-hot 指向中心行 pnt.link_v

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

