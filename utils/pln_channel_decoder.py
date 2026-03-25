from __future__ import annotations

from typing import Dict

import torch


def decode_branch_channels_logits(
    pred: torch.Tensor,
    *,
    s: int = 14,
    num_classes: int = 20,
    num_points: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Split channels into raw logits/scores ONLY (no sigmoid/softmax).
    This is suitable for loss computation.
    """
    if pred.dim() != 4:
        raise ValueError(f"pred must be (B,C,S,S). Got shape={tuple(pred.shape)}")
    b, c, h, w = pred.shape
    if h != s or w != s:
        raise ValueError(f"Expected spatial size (S,S)=({s},{s}), got ({h},{w})")

    expected_c = num_points * (1 + num_classes + 2 + 2 * s)
    if c != expected_c:
        raise ValueError(f"Expected C={expected_c} (=num_points*51), got C={c}")

    per_point = 1 + num_classes + 2 + 2 * s
    x = pred.view(b, num_points, per_point, s, s)

    p_logits = x[:, :, 0, :, :]  # (B,P,S,S)
    q_logits = x[:, :, 1 : 1 + num_classes, :, :]  # (B,P,N,S,S)
    dx_logits = x[:, :, 1 + num_classes, :, :]  # (B,P,S,S)
    dy_logits = x[:, :, 1 + num_classes + 1, :, :]  # (B,P,S,S)

    links_logits = x[:, :, 1 + num_classes + 2 : 1 + num_classes + 2 + 2 * s, :, :]  # (B,P,2S,S,S)
    lx_logits = links_logits[:, :, 0:s, :, :]  # (B,P,S,S,S)
    ly_logits = links_logits[:, :, s : 2 * s, :, :]  # (B,P,S,S,S)

    return {
        "P_logits": p_logits,
        "Q_logits": q_logits,
        "dx_logits": dx_logits,
        "dy_logits": dy_logits,
        "Lx_logits": lx_logits,
        "Ly_logits": ly_logits,
    }


def decode_branch_channels_inference(
    pred: torch.Tensor,
    *,
    s: int = 14,
    num_classes: int = 20,
    num_points: int = 4,
    dxdy_range: tuple[float, float] = (0.0, 1.0),
) -> Dict[str, torch.Tensor]:
    """
    Split channels and then normalize into inference-friendly outputs:
    sigmoid/softmax for P/Q/Lx/Ly and [dx,dy] mapping into dxdy_range.
    """
    logits = decode_branch_channels_logits(
        pred, s=s, num_classes=num_classes, num_points=num_points
    )

    p = torch.sigmoid(logits["P_logits"])
    q = torch.softmax(logits["Q_logits"], dim=2)

    dx = torch.sigmoid(logits["dx_logits"])
    dy = torch.sigmoid(logits["dy_logits"])
    lo, hi = dxdy_range
    if (lo, hi) != (0.0, 1.0):
        dx = dx * (hi - lo) + lo
        dy = dy * (hi - lo) + lo

    # Softmax over k dimension.
    Lx = torch.softmax(logits["Lx_logits"], dim=2)
    Ly = torch.softmax(logits["Ly_logits"], dim=2)

    return {
        "P": p,
        "Q": q,
        "dx": dx,
        "dy": dy,
        "Lx": Lx,
        "Ly": Ly,
        **logits,
    }


def decode_branch_channels(
    pred: torch.Tensor,
    *,
    s: int = 14,
    num_classes: int = 20,
    num_points: int = 4,
    dxdy_range: tuple[float, float] = (0.0, 1.0),
) -> Dict[str, torch.Tensor]:
    """Backward-compatible inference decoder (normalized)."""
    return decode_branch_channels_inference(
        pred,
        s=s,
        num_classes=num_classes,
        num_points=num_points,
        dxdy_range=dxdy_range,
    )

