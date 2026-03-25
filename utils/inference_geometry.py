from __future__ import annotations

from typing import Tuple, Union, Optional

import torch

Number = Union[int, float]


def uv_offset_to_image_xy(
    u: Union[int, torch.Tensor],
    v: Union[int, torch.Tensor],
    dx: Union[float, torch.Tensor],
    dy: Union[float, torch.Tensor],
    stride: Union[int, float],
    *,
    clamp_offsets: bool = True,
    offsets_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map (u,v) grid cell index + normalized offsets (dx,dy) to original image coordinates.

    Current convention (right-top corner as reference):
      Let the cell right-top corner be:
        x_rt = (u + 1) * stride
        y_rt = v * stride

      Then offsets are:
        dx = (x_rt - x) / stride
        dy = (y - y_rt) / stride

      So inverse mapping is:
        x = (u + 1 - dx) * stride
        y = (v + dy) * stride

    Where dx/dy are normalized relative to one cell size, typically in [0,1].
    Coordinate convention:
      - x: left -> right
      - y: top -> bottom

    Args:
        u, v: grid indices on feature map, can be int or tensors.
        dx, dy: offsets relative to the cell, typically in [0,1].
        stride: pixels per feature-map step.

    Returns:
        (x, y) as torch tensors (broadcasted).
    """
    stride_t = torch.as_tensor(stride, dtype=torch.float32)

    u_t = torch.as_tensor(u, dtype=torch.float32, device=dx.device if isinstance(dx, torch.Tensor) else None)
    v_t = torch.as_tensor(v, dtype=torch.float32, device=dy.device if isinstance(dy, torch.Tensor) else None)

    dx_t = torch.as_tensor(dx, dtype=torch.float32)
    dy_t = torch.as_tensor(dy, dtype=torch.float32)

    if clamp_offsets:
        lo, hi = offsets_range
        dx_t = dx_t.clamp(min=lo, max=hi)
        dy_t = dy_t.clamp(min=lo, max=hi)

    # Broadcast then compute absolute coordinates (right-top referenced offsets)
    x = (u_t + 1.0 - dx_t) * stride_t
    y = (v_t + dy_t) * stride_t
    return x, y


def box_from_corner_and_center(
    corner_xy: Union[Tuple[Number, Number], torch.Tensor],
    center_xy: Union[Tuple[Number, Number], torch.Tensor],
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[float, float, float, float]]:
    """
    Given one corner point and the center point (in original image coordinates),
    return the axis-aligned box (x_min, y_min, x_max, y_max).

    IMPORTANT:
    The vector (corner - center) is a HALF diagonal, not a full diagonal.
    We first recover the opposite corner by center symmetry:
      opposite = 2 * center - corner
    Then take min/max between (corner, opposite).
    """
    # Tensor path
    if isinstance(corner_xy, torch.Tensor) or isinstance(center_xy, torch.Tensor):
        corner = torch.as_tensor(corner_xy, dtype=torch.float32)
        center = torch.as_tensor(center_xy, dtype=torch.float32)
        corner_x, corner_y = corner[..., 0], corner[..., 1]
        center_x, center_y = center[..., 0], center[..., 1]

        opposite_x = 2.0 * center_x - corner_x
        opposite_y = 2.0 * center_y - corner_y

        x_min = torch.minimum(corner_x, opposite_x)
        x_max = torch.maximum(corner_x, opposite_x)
        y_min = torch.minimum(corner_y, opposite_y)
        y_max = torch.maximum(corner_y, opposite_y)
        return x_min, y_min, x_max, y_max

    # Tuple/number path
    cx, cy = corner_xy  # type: ignore[misc]
    mx, my = center_xy  # type: ignore[misc]
    ox = 2.0 * mx - cx
    oy = 2.0 * my - cy
    x_min = min(cx, ox)
    x_max = max(cx, ox)
    y_min = min(cy, oy)
    y_max = max(cy, oy)
    return x_min, y_min, x_max, y_max

