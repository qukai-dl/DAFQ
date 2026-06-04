"""
Cross-Task Dependency-Aware Alignment Module  (Sec. 3.2 of DAFQ-Net)

This is the central architectural innovation of the paper.  It performs
three sequential operations:

    1) **Task-specific encoding** of future exogenous inputs.  For every
       task i, a task-specific MLP maps the flattened future features
       X^Fut_i (shape (B, H, D_i)) into a fixed-size context vector
       c^i in R^{d_fut}.  The three vectors are stacked into a "task
       matrix"  C in R^{3 x d_fut}.

    2) **Multi-Head Self-Attention across tasks.**  Treating the 3 tasks
       as the sequence dimension, MHSA explicitly computes the inter-
       task affinity matrix, providing a direct mathematical quantifi-
       cation of the dependencies between heterogeneous energy streams.
       Residual connection + LayerNorm yield the enhanced context
       C_tilde.

    3) **Dependency-aware adaptive fusion.**  A sigmoid gate g^i is
       learned from the concatenation of the global historical context
       h_global and the task-specific future context c_tilde^i.  The
       final task latent is

           z^i = g^i * h_global + (1 - g^i) * c_tilde^i,

       which is a *weighted* combination instead of a naive concat.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Task-specific future-context encoder
# ----------------------------------------------------------------------
class TaskFutureEncoder(nn.Module):
    """
    Flatten  +  MLP  (ReLU)  per task.

    Input  : (B, H, D_i)
    Output : (B, d_fut)
    """

    def __init__(self, horizon: int, feat_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(horizon * feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------
# Cross-Task Multi-Head Self-Attention
# ----------------------------------------------------------------------
class CrossTaskAttention(nn.Module):
    """
    Multi-Head Self-Attention applied over the *task* dimension.

    For a batch of task matrices  C in R^{B x 3 x d_fut}, each row of C
    is treated as one "token".  The output has the same shape
    (B, 3, d_fut) (we set d_v = d_fut for dimension consistency, as
    stated in the paper).
    """

    def __init__(self, d_fut: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_fut % num_heads == 0, "d_fut must be divisible by num_heads"
        self.attn = nn.MultiheadAttention(
            embed_dim=d_fut,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_fut)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: (B, 3, d_fut)  task matrix.
        Returns:
            (B, 3, d_fut)    enhanced task matrix after attention + residual.
        """
        attn_out, _ = self.attn(c, c, c, need_weights=False)
        return self.norm(c + attn_out)


# ----------------------------------------------------------------------
# Dependency-aware adaptive fusion gate
# ----------------------------------------------------------------------
class DependencyAwareFusion(nn.Module):
    """
    For each task i, computes a fusion gate

        g^i = sigmoid( MLP_fusion( h_global  ||  c_tilde^i ) )

    and outputs the gated latent

        z^i = g^i * h_global  +  (1 - g^i) * c_tilde^i.

    Because g^i has the same shape as h_global and c_tilde^i (both equal
    to d_fut — see ``_align_dims``), the combination is element-wise
    rather than concatenative.
    """

    def __init__(self, d_hist: int, d_fut: int, hidden: int) -> None:
        super().__init__()
        # The gate MLP works on the concatenation of h_global and c_tilde^i.
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_hist + d_fut, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_fut),  # produce a gate of dim d_fut
        )

    def forward(
        self,
        h_global: torch.Tensor,   # (B, d_hist)
        c_tilde: torch.Tensor,    # (B, 3, d_fut)
    ) -> torch.Tensor:
        # Broadcast h_global over the 3 tasks to enable concatenation.
        h_exp = h_global.unsqueeze(1).expand(-1, c_tilde.size(1), -1)   # (B,3,d_hist)
        cat = torch.cat([h_exp, c_tilde], dim=-1)                      # (B,3,d_hist+d_fut)
        g = torch.sigmoid(self.gate_mlp(cat))                          # (B,3,d_fut)
        return g * h_exp + (1.0 - g) * c_tilde                         # (B,3,d_fut)


# ----------------------------------------------------------------------
# Full Cross-Task Dependency-Aware Alignment block
# ----------------------------------------------------------------------
class CrossTaskAlignment(nn.Module):
    """
    Combines the three sub-modules above.

    The input is a dict of future-exogenous tensors

        fut_feats = {
            "load": (B, H, D_load),
            "wind": (B, H, D_wind),
            "pv":   (B, H, D_pv),
        }

    and the output is a dict of task latents

        z = {
            "load": (B, d_fut),
            "wind": (B, d_fut),
            "pv":   (B, d_fut),
        }
    """

    TASK_NAMES = ("load", "wind", "pv")

    def __init__(
        self,
        horizon: int,
        feat_dims: dict[str, int],   # per-task future feature dim
        d_fut: int = 128,
        d_hist: int | None = None,   # dim of h_global; defaults to d_fut
        num_heads: int = 4,
        dropout: float = 0.1,
        gate_hidden: int = 128,
    ) -> None:
        super().__init__()
        d_hist = d_hist or d_fut

        # 1) per-task MLP encoders
        self.task_encoders = nn.ModuleDict({
            name: TaskFutureEncoder(
                horizon=horizon,
                feat_dim=feat_dims[name],
                hidden=d_fut,
                out_dim=d_fut,
            )
            for name in self.TASK_NAMES
        })

        # 2) cross-task MHSA
        self.cross_attn = CrossTaskAttention(
            d_fut=d_fut,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 3) dependency-aware adaptive fusion
        self.fusion = DependencyAwareFusion(
            d_hist=d_hist,
            d_fut=d_fut,
            hidden=gate_hidden,
        )

    def forward(
        self,
        h_global: torch.Tensor,                       # (B, d_hist)
        fut_feats: dict[str, torch.Tensor],           # see class doc
    ) -> dict[str, torch.Tensor]:
        # Step 1: per-task encoding -> task matrix C
        c_list = [self.task_encoders[name](fut_feats[name]) for name in self.TASK_NAMES]
        c = torch.stack(c_list, dim=1)                                 # (B, 3, d_fut)

        # Step 2: cross-task attention
        c_tilde = self.cross_attn(c)                                   # (B, 3, d_fut)

        # Step 3: adaptive fusion with global historical context
        z = self.fusion(h_global, c_tilde)                             # (B, 3, d_fut)

        return {name: z[:, i, :] for i, name in enumerate(self.TASK_NAMES)}
