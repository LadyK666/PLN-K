from __future__ import annotations

from typing import Dict, Literal, Tuple

import torch


BranchName = Literal["left_top", "right_top", "left_bottom", "right_bottom"]


def generate_candidate_pairs_from_links(
    P: torch.Tensor,
    Lx: torch.Tensor,
    Ly: torch.Tensor,
    *,
    branch: BranchName,
    s: int,
    num_points: int = 4,
    B_point: int = 2,
    p_threshold: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Generate candidate point pairs from (Lx, Ly) links for one branch.

    Input tensors come from `decode_branch_channels()` for a single branch:
      P  : (B, 4, S, S)  - existence probability per point type
      Lx : (B, 4, S, S, S) where Lx[b,p,k,v,u] is prob linking to row index k
      Ly : (B, 4, S, S, S) where Ly[b,p,k,v,u] is prob linking to col index k

    For each source point (b, p, v, u):
      - u_row = argmax_k Lx[b,p,:,v,u]
      - u_col = argmax_k Ly[b,p,:,v,u]
      - linked target cell is (v_t=u_row, u_t=u_col)

    Point type mapping:
      - j in {1,2} are "center"
      - j in {3,4} are "corner"
      - with B_point=2 => target point type t satisfies |t-j|=B_point
        i.e. center(0..1) -> corner(2..3), corner(2..3) -> center(0..1)

    Branch geometry filtering:
      - left_top:     corner is left-top of center  (corner_x < center_x, corner_y < center_y)
      - right_top:    corner is right-top of center (corner_x > center_x, corner_y < center_y)
      - left_bottom:  corner is left-bottom of center (corner_x < center_x, corner_y > center_y)
      - right_bottom: corner is right-bottom of center (corner_x > center_x, corner_y > center_y)

    Returns:
      dict with variable tensors to help later loss/box computation:
        - pairs_ijst: (N,4) columns are [i, j, s_diag, t]
            where i = src_x + src_y (diagonal sum index)
                  j = src point type in {1..4}
                  s_diag = tgt_x + tgt_y (diagonal sum index)
                  t = tgt point type in {1..4}
        - src_xy: (N,2) [src_x(u), src_y(v)]
        - tgt_xy: (N,2) [tgt_x(u), tgt_y(v)]
        - src_point_type: (N,) in {0..3}
        - tgt_point_type: (N,) in {0..3}
        - center_xy: (N,2) center point coord after mapping
        - corner_xy: (N,2) corner point coord after mapping
            - center_point_type: (N,) center point type idx in {0..3}
            - corner_point_type: (N,) corner point type idx in {0..3}
        - batch_idx: (N,)
        - score: (N,) placeholder confidence computed as P_src * P_tgt
    """
    if P.dim() != 4:
        raise ValueError(f"P must be (B,4,S,S), got {tuple(P.shape)}")
    if Lx.dim() != 5 or Ly.dim() != 5:
        raise ValueError(f"Lx/Ly must be (B,4,S,S,S), got {tuple(Lx.shape)}, {tuple(Ly.shape)}")

    bsz, p_dim, s_y, s_x = P.shape
    if p_dim != num_points:
        raise ValueError(f"Expected num_points={num_points}, got P shape {tuple(P.shape)}")
    if s_y != s or s_x != s:
        raise ValueError(f"Expected S={s} for spatial, got P spatial {(s_y,s_x)}")

    if Lx.shape[2] != s or Lx.shape[3] != s or Lx.shape[4] != s:
        raise ValueError(f"Expected Lx shape (B,{num_points},{s},{s},{s}), got {tuple(Lx.shape)}")

    device = P.device

    # Compute argmax indices from links.
    # 根据你的约定：
    #   argmax_k L(k)x_ij = u_t  (对应水平/列索引)
    #   argmax_k L(k)y_ij = v_t  (对应竖直/行索引)
    # 注意张量布局：src点用 pred[..., v, u]，因此 P/Lx/Ly 的最后两维分别是 (v,u)。
    # Lx/Ly 形状：(B,P,k,S,S)，argmax 在 k 维上得到目标的 (u_t 或 v_t)。
    tgt_u = Lx.argmax(dim=2)  # (B,4,S,S) -> target u (x/col)
    tgt_v = Ly.argmax(dim=2)  # (B,4,S,S) -> target v (y/row)

    # Grid coords for src cell
    u_coords = torch.arange(s, device=device)
    v_coords = torch.arange(s, device=device)
    grid_u = u_coords.view(1, s).repeat(s, 1)  # (v,u) -> u
    grid_v = v_coords.view(s, 1).repeat(1, s)  # (v,u) -> v

    grid_u = grid_u.view(1, 1, s, s).expand(bsz, num_points, s, s)
    grid_v = grid_v.view(1, 1, s, s).expand(bsz, num_points, s, s)

    # Point type index (0..3)
    p_idx = torch.arange(num_points, device=device).view(1, num_points, 1, 1).expand(bsz, num_points, s, s)
    src_is_center = p_idx < B_point  # centers are j in [1,2] => p in [0,1]

    # Target point type from |t-j|=B_point
    # if src is center -> tgt is corner (p+2); else -> tgt is center (p-2)
    tgt_p_idx = torch.where(src_is_center, p_idx + B_point, p_idx - B_point)

    # Determine center/corner coordinates depending on which point type the source is.
    # If src is center: center at (src_u,src_v), corner at (tgt_u,tgt_v)
    # If src is corner: corner at (src_u,src_v), center at (tgt_u,tgt_v)
    center_x = torch.where(src_is_center, grid_u, tgt_u)
    center_y = torch.where(src_is_center, grid_v, tgt_v)
    corner_x = torch.where(src_is_center, tgt_u, grid_u)
    corner_y = torch.where(src_is_center, tgt_v, grid_v)

    # Branch filtering by relative position.
    if branch == "left_top":
        mask = (corner_x < center_x) & (corner_y < center_y)
    elif branch == "right_top":
        mask = (corner_x > center_x) & (corner_y < center_y)
    elif branch == "left_bottom":
        mask = (corner_x < center_x) & (corner_y > center_y)
    elif branch == "right_bottom":
        mask = (corner_x > center_x) & (corner_y > center_y)
    else:
        raise ValueError(f"Unknown branch={branch}")

    # Optional existence gating by P_src.
    if p_threshold > 0.0:
        mask = mask & (P > p_threshold)

    # Extract candidate indices.
    idx = mask.nonzero(as_tuple=False)  # (N,4): b,p,v,u
    if idx.numel() == 0:
        # Return empty tensors with correct dtypes.
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return {
            "pairs_ijst": torch.empty((0, 4), device=device, dtype=torch.long),
            "src_xy": torch.empty((0, 2), device=device, dtype=torch.float32),
            "tgt_xy": torch.empty((0, 2), device=device, dtype=torch.float32),
            "src_point_type": empty,
            "tgt_point_type": empty,
            "center_xy": torch.empty((0, 2), device=device, dtype=torch.float32),
            "corner_xy": torch.empty((0, 2), device=device, dtype=torch.float32),
            "batch_idx": empty,
            "score": torch.empty((0,), device=device, dtype=torch.float32),
        }

    batch_idx = idx[:, 0]
    src_p = idx[:, 1]
    v_src = idx[:, 2]
    u_src = idx[:, 3]

    # Gather src/tgt coords
    src_x_sel = grid_u[batch_idx, src_p, v_src, u_src]
    src_y_sel = grid_v[batch_idx, src_p, v_src, u_src]
    tgt_x_sel = tgt_u[batch_idx, src_p, v_src, u_src]  # u_t
    tgt_y_sel = tgt_v[batch_idx, src_p, v_src, u_src]  # v_t

    # Diagonal-sum indices
    i_diag = src_x_sel + src_y_sel
    s_diag = tgt_x_sel + tgt_y_sel

    # Point types in {1..4}
    j_point = src_p + 1
    tgt_p_sel = tgt_p_idx[batch_idx, src_p, v_src, u_src]
    t_point = tgt_p_sel + 1

    # Center/corner coords
    center_x_sel = center_x[batch_idx, src_p, v_src, u_src]
    center_y_sel = center_y[batch_idx, src_p, v_src, u_src]
    corner_x_sel = corner_x[batch_idx, src_p, v_src, u_src]
    corner_y_sel = corner_y[batch_idx, src_p, v_src, u_src]

    # Center/corner point type idx (0..3)
    center_p_sel = torch.where(src_is_center[batch_idx, src_p, v_src, u_src], src_p, tgt_p_sel)
    corner_p_sel = torch.where(src_is_center[batch_idx, src_p, v_src, u_src], tgt_p_sel, src_p)

    # Confidence score placeholder: P_src * P_tgt
    p_src = P[batch_idx, src_p, v_src, u_src]
    p_tgt = P[batch_idx, tgt_p_sel, tgt_y_sel, tgt_x_sel]
    score = p_src * p_tgt

    pairs_ijst = torch.stack([i_diag, j_point, s_diag, t_point], dim=1).to(torch.long)

    return {
        "pairs_ijst": pairs_ijst,
        "src_xy": torch.stack([src_x_sel, src_y_sel], dim=1).to(torch.float32),
        "tgt_xy": torch.stack([tgt_x_sel, tgt_y_sel], dim=1).to(torch.float32),
        "src_point_type": src_p.to(torch.long),
        "tgt_point_type": tgt_p_sel.to(torch.long),
        "center_xy": torch.stack([center_x_sel, center_y_sel], dim=1).to(torch.float32),
        "corner_xy": torch.stack([corner_x_sel, corner_y_sel], dim=1).to(torch.float32),
        "center_point_type": center_p_sel.to(torch.long),
        "corner_point_type": corner_p_sel.to(torch.long),
        "batch_idx": batch_idx.to(torch.long),
        "score": score.to(torch.float32),
    }

