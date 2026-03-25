from __future__ import annotations

from typing import Dict, Literal, Tuple

import torch


def compute_pair_object_probability(
    *,
    P: torch.Tensor,
    Q: torch.Tensor,
    Lx: torch.Tensor,
    Ly: torch.Tensor,
    pairs_dict: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute per-pair per-class object probability:

      P_obj_ijnst^{(n)} =
        P_ij * P_st * Q_ij^{(n)} * Q_st^{(n)} *
        ( Lx_ij^{(s_x)} * Ly_ij^{(s_y)} + Lx_st^{(i_x)} * Ly_st^{(i_y)} ) / 2

    where (i_x,i_y) is source grid cell (u,v), (s_x,s_y) is target grid cell (u_t,v_t).

    Args:
        P : (B,4,S,S) normalized existence probability for each point type j (0..3)
        Q : (B,4,N,S,S) normalized class probabilities for each point type j (0..3)
        Lx: (B,4,S,S,S) normalized link distribution for linked u/x/col index k
        Ly: (B,4,S,S,S) normalized link distribution for linked v/y/row index k
        pairs_dict: output of `generate_candidate_pairs_from_links()`, must include:
            - batch_idx: (Np,)
            - src_point_type: (Np,)  in {0..3}
            - tgt_point_type: (Np,)  in {0..3}
            - src_xy: (Np,2) [u_src, v_src] as integer-like values (float ok)
            - tgt_xy: (Np,2) [u_tgt, v_tgt] as integer-like values (float ok)

    Returns:
        prob: (Np, N) probability that each candidate pair corresponds to class n.
    """
    if P.dim() != 4:
        raise ValueError(f"P must be (B,4,S,S). Got {tuple(P.shape)}")
    if Q.dim() != 5:
        raise ValueError(f"Q must be (B,4,N,S,S). Got {tuple(Q.shape)}")
    if Lx.dim() != 5 or Ly.dim() != 5:
        raise ValueError(f"Lx/Ly must be (B,4,S,S,S). Got {tuple(Lx.shape)}, {tuple(Ly.shape)}")

    bP, p4, s1, s2 = P.shape
    bQ, p4b, n_classes, sQ1, sQ2 = Q.shape
    bL, p4c, sL1, sL2, sL3 = Lx.shape
    if not (bP == bQ == bL and p4 == p4b == p4c and s1 == s2 == sQ1 == sQ2 and sL1 == sL2 == sL3 == s1):
        raise ValueError("Shape mismatch among P/Q/Lx/Ly")

    Np = pairs_dict["batch_idx"].shape[0]
    device = P.device

    batch_idx = pairs_dict["batch_idx"].to(device=device, dtype=torch.long)  # (Np,)
    src_p = pairs_dict["src_point_type"].to(device=device, dtype=torch.long)  # (Np,)
    tgt_p = pairs_dict["tgt_point_type"].to(device=device, dtype=torch.long)  # (Np,)

    src_xy = pairs_dict["src_xy"].to(device=device)
    tgt_xy = pairs_dict["tgt_xy"].to(device=device)
    u_src = src_xy[:, 0].round().to(torch.long)
    v_src = src_xy[:, 1].round().to(torch.long)
    u_tgt = tgt_xy[:, 0].round().to(torch.long)
    v_tgt = tgt_xy[:, 1].round().to(torch.long)

    # Existence terms
    p_src = P[batch_idx, src_p, v_src, u_src]  # (Np,)
    p_tgt = P[batch_idx, tgt_p, v_tgt, u_tgt]  # (Np,)

    # Class terms
    q_src = Q[batch_idx, src_p, :, v_src, u_src]  # (Np,N)
    q_tgt = Q[batch_idx, tgt_p, :, v_tgt, u_tgt]  # (Np,N)

    # Link consistency terms.
    # Source -> target: Lx_ij^{(s_x)} * Ly_ij^{(s_y)}
    lx_src = Lx[batch_idx, src_p, u_tgt, v_src, u_src]  # (Np,)
    ly_src = Ly[batch_idx, src_p, v_tgt, v_src, u_src]  # (Np,)
    link_src_to_tgt = lx_src * ly_src

    # Target -> source: Lx_st^{(i_x)} * Ly_st^{(i_y)}
    lx_tgt = Lx[batch_idx, tgt_p, u_src, v_tgt, u_tgt]  # (Np,)
    ly_tgt = Ly[batch_idx, tgt_p, v_src, v_tgt, u_tgt]  # (Np,)
    link_tgt_to_src = lx_tgt * ly_tgt

    link_avg = 0.5 * (link_src_to_tgt + link_tgt_to_src)  # (Np,)

    # Assemble final per-class probabilities
    prob = (p_src * p_tgt * link_avg).unsqueeze(1) * q_src * q_tgt  # (Np,N)
    return prob


def compute_pair_scores_max_n(
    *,
    P: torch.Tensor,
    Q: torch.Tensor,
    Lx: torch.Tensor,
    Ly: torch.Tensor,
    pairs_dict: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute per-pair confidence score as max over class:
      score_ijst = max_n P^{obj}_{ijnst}(n)

    Returns:
      score: (Np,)
    """
    prob = compute_pair_object_probability(P=P, Q=Q, Lx=Lx, Ly=Ly, pairs_dict=pairs_dict)  # (Np,N)
    score = prob.max(dim=1).values  # (Np,)
    return score


def compute_pair_scores_and_labels_max_n(
    *,
    P: torch.Tensor,
    Q: torch.Tensor,
    Lx: torch.Tensor,
    Ly: torch.Tensor,
    pairs_dict: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-pair:
      score = max_n P_obj(n)
      label = argmax_n P_obj(n)

    Returns:
      score: (Np,)
      labels: (Np,) in [0, N-1]
    """
    prob = compute_pair_object_probability(P=P, Q=Q, Lx=Lx, Ly=Ly, pairs_dict=pairs_dict)  # (Np,N)
    score, labels = prob.max(dim=1)
    return score, labels


def attach_pair_scores_max_n(
    *,
    P: torch.Tensor,
    Q: torch.Tensor,
    Lx: torch.Tensor,
    Ly: torch.Tensor,
    pairs_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Convenience wrapper: computes score=max_n(prob) and writes it to pairs_dict["score"].
    """
    pairs_dict = dict(pairs_dict)
    pairs_dict["score"] = compute_pair_scores_max_n(P=P, Q=Q, Lx=Lx, Ly=Ly, pairs_dict=pairs_dict)
    return pairs_dict


def attach_pair_scores_and_labels_max_n(
    *,
    P: torch.Tensor,
    Q: torch.Tensor,
    Lx: torch.Tensor,
    Ly: torch.Tensor,
    pairs_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Convenience wrapper: computes score=max_n(prob), label=argmax_n(prob)
    and writes them into pairs_dict["score"], pairs_dict["label"].
    """
    pairs_dict = dict(pairs_dict)
    score, labels = compute_pair_scores_and_labels_max_n(
        P=P, Q=Q, Lx=Lx, Ly=Ly, pairs_dict=pairs_dict
    )
    pairs_dict["score"] = score
    pairs_dict["label"] = labels.to(torch.long)
    return pairs_dict

