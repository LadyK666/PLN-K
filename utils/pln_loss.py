from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class PLNLossWeights:
    w_class: float = 1.0
    w_coord: float = 1.0
    w_link: float = 1.0


class PLNLoss(nn.Module):
    """
    PLN point-wise loss based on the provided equations.

    Definitions (per point ij):
      Loss_ij = 1_pt_ij * Loss_pt_ij + 1_nopt_ij * Loss_nopt_ij

      Loss_pt_ij =
          (P_ij - 1)^2
          + w_class * sum_n (Q_ij^n - Qhat_ij^n)^2
          + w_coord * ((x_ij - xhat_ij)^2 + (y_ij - yhat_ij)^2)
          + w_link  * sum_k ((Lx_ij^k - Lxhat_ij^k)^2 + (Ly_ij^k - Lyhat_ij^k)^2)

      Loss_nopt_ij = P_ij^2

    Total:
      Loss = sum_{i=1..S^2} sum_{j=1..2B} Loss_ij

    Notes:
    - This module expects decoded (normalized) tensors, e.g. output from
      `decode_branch_channels_inference`.
    - The point/non-point mask generation logic is intentionally left as TODO.
      Please pass `pt_mask` and `nopt_mask` explicitly for now.
    """

    def __init__(
        self,
        weights: Optional[PLNLossWeights] = None,
        reduction: str = "mean",
        logit_clip: Optional[float] = 20.0,
    ):
        super().__init__()
        self.weights = weights if weights is not None else PLNLossWeights()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction={reduction}")
        self.reduction = reduction
        self.logit_clip = logit_clip

    def _build_point_masks_todo(self, target: Dict[str, torch.Tensor]):
        """
        TODO:
        Implement logic for:
          - 1_pt_ij   (point exists)
          - 1_nopt_ij (no point)
        based on dataset annotations / matching strategy.
        """
        raise NotImplementedError(
            "Point/non-point mask generation is TODO. "
            "Pass pt_mask and nopt_mask explicitly to PLNLoss.forward()."
        )

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        *,
        pt_mask: Optional[torch.Tensor] = None,
        nopt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred:
              - P:  (B,P,S,S)
              - Q:  (B,P,N,S,S)
              - dx or x: (B,P,S,S)
              - dy or y: (B,P,S,S)
              - Lx: (B,P,S,S,S)
              - Ly: (B,P,S,S,S)
            target: same key structure as pred, using hat variables
            pt_mask:   (B,P,S,S) with 0/1 values for points
            nopt_mask: (B,P,S,S) with 0/1 values for no-points

        Returns:
            dict with total loss and component losses.
        """
        P = pred["P"]
        Q = pred["Q"]
        x = pred["x"] if "x" in pred else pred["dx"]
        y = pred["y"] if "y" in pred else pred["dy"]
        Lx = pred["Lx"]
        Ly = pred["Ly"]

        P_hat = target["P"]
        Q_hat = target["Q"]
        x_hat = target["x"] if "x" in target else target["dx"]
        y_hat = target["y"] if "y" in target else target["dy"]
        Lx_hat = target["Lx"]
        Ly_hat = target["Ly"]

        # Numerical stabilization for raw-logit regression/classification:
        # still uses unnormalized outputs, but prevents overflow in squared loss.
        if self.logit_clip is not None:
            c = float(self.logit_clip)
            P = P.clamp(min=-c, max=c)
            Q = Q.clamp(min=-c, max=c)
            x = x.clamp(min=-c, max=c)
            y = y.clamp(min=-c, max=c)
            Lx = Lx.clamp(min=-c, max=c)
            Ly = Ly.clamp(min=-c, max=c)

        if pt_mask is None or nopt_mask is None:
            self._build_point_masks_todo(target)

        # Convert masks to float for arithmetic.
        pt_mask = pt_mask.float()
        nopt_mask = nopt_mask.float()

        # (P_ij - 1)^2
        p_pos = (P - 1.0) ** 2  # (B,P,S,S)
        # sum_n (Q - Qhat)^2
        q_pos = ((Q - Q_hat) ** 2).sum(dim=2)  # (B,P,S,S)
        # (x-xhat)^2 + (y-yhat)^2
        coord_pos = (x - x_hat) ** 2 + (y - y_hat) ** 2  # (B,P,S,S)
        # sum_k ((Lx-Lxhat)^2 + (Ly-Lyhat)^2)
        link_pos = ((Lx - Lx_hat) ** 2 + (Ly - Ly_hat) ** 2).sum(dim=2)  # (B,P,S,S)

        loss_pt = p_pos + self.weights.w_class * q_pos + self.weights.w_coord * coord_pos + self.weights.w_link * link_pos
        loss_nopt = P ** 2

        # Loss_ij = 1_pt * loss_pt + 1_nopt * loss_nopt
        loss_ij = pt_mask * loss_pt + nopt_mask * loss_nopt

        # Unweighted/weighted positive components (masked by pt) for diagnostics.
        p_pt = pt_mask * p_pos
        q_pt_raw = pt_mask * q_pos
        coord_pt_raw = pt_mask * coord_pos
        link_pt_raw = pt_mask * link_pos
        q_pt_w = self.weights.w_class * q_pt_raw
        coord_pt_w = self.weights.w_coord * coord_pt_raw
        link_pt_w = self.weights.w_link * link_pt_raw
        if self.reduction == "sum":
            total = loss_ij.sum()
            pt_term = (pt_mask * loss_pt).sum()
            nopt_term = (nopt_mask * loss_nopt).sum()
            p_pt_term = p_pt.sum()
            q_pt_raw_term = q_pt_raw.sum()
            coord_pt_raw_term = coord_pt_raw.sum()
            link_pt_raw_term = link_pt_raw.sum()
            q_pt_w_term = q_pt_w.sum()
            coord_pt_w_term = coord_pt_w.sum()
            link_pt_w_term = link_pt_w.sum()
        else:
            # mean over batch (sum over point/grid dims)
            B = loss_ij.shape[0]
            total = loss_ij.view(B, -1).sum(dim=1).mean()
            pt_term = (pt_mask * loss_pt).view(B, -1).sum(dim=1).mean()
            nopt_term = (nopt_mask * loss_nopt).view(B, -1).sum(dim=1).mean()
            p_pt_term = p_pt.view(B, -1).sum(dim=1).mean()
            q_pt_raw_term = q_pt_raw.view(B, -1).sum(dim=1).mean()
            coord_pt_raw_term = coord_pt_raw.view(B, -1).sum(dim=1).mean()
            link_pt_raw_term = link_pt_raw.view(B, -1).sum(dim=1).mean()
            q_pt_w_term = q_pt_w.view(B, -1).sum(dim=1).mean()
            coord_pt_w_term = coord_pt_w.view(B, -1).sum(dim=1).mean()
            link_pt_w_term = link_pt_w.view(B, -1).sum(dim=1).mean()

        return {
            "loss": total,
            "loss_pt": pt_term,
            "loss_nopt": nopt_term,
            "loss_pt_p": p_pt_term,
            "loss_pt_q_raw": q_pt_raw_term,
            "loss_pt_coord_raw": coord_pt_raw_term,
            "loss_pt_link_raw": link_pt_raw_term,
            "loss_pt_q_weighted": q_pt_w_term,
            "loss_pt_coord_weighted": coord_pt_w_term,
            "loss_pt_link_weighted": link_pt_w_term,
        }

