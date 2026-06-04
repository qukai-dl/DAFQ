"""
DAFQ-Net : Dependency-Aware Fusion Quantile Network
====================================================

End-to-end PyTorch implementation of the DAFQ-Net model proposed in:

    "Operation-Oriented Dependency-Aware Joint Probabilistic Forecasting
     of Load and Renewable Generation for Microgrid Dispatch"

Architecture (Sec. 3 of the paper):

    Historical context
        └─► Shared Context Encoder (Transformer)          → h_global
    Future exogenous context
        └─► Task-specific MLP encoders                    → C  (task matrix)
                └─► Cross-Task Multi-Head Self-Attention  → C_tilde
                        └─► Dependency-Aware Fusion       → {z^Load, z^Wind, z^PV}
    Quantile decoder (per task)
        └─► Linear head  →  V^i
                └─► Hadamard( Physics gate G^i )          → Y^i_hat
                └─► ReLU

    Loss
        └─► Homoscedastic Uncertainty-Aware Multi-Task Pinball Loss

This file wires the four building blocks (shared_encoder,
cross_task_fusion, quantile_decoder, losses) into a single ``nn.Module``.
The model is intentionally agnostic to data loading, training loops, and
sampling scenarios — only the core architectural design is exposed.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from shared_encoder import SharedContextEncoder
from cross_task_fusion import CrossTaskAlignment
from quantile_decoder import MultiTaskQuantileDecoder
from losses import UncertaintyAwareMultiTaskLoss


# ----------------------------------------------------------------------
# Default quantile set used in the paper
# ----------------------------------------------------------------------
DEFAULT_QUANTILES: tuple[float, ...] = tuple(i / 20.0 for i in range(1, 20))   # 0.05 ... 0.95


# ----------------------------------------------------------------------
# Main model
# ----------------------------------------------------------------------
class DAFQNet(nn.Module):
    """
    Dependency-Aware Fusion Quantile Network.

    Args:
        look_back:           historical window length L.
        horizon:             forecasting horizon H.
        n_targets:           number of target variables (default 3:
                             Load, Wind, PV).
        common_feat_dim:     D_c, dim of common exogenous features.
        future_feat_dims:    dict  {"load": D_load, "wind": D_wind, "pv": D_pv}.
        d_model:             Transformer model dimension.
        nhead:               number of attention heads in the shared encoder.
        num_encoder_layers:  number of Transformer encoder layers (M).
        d_fut:               dim of the task-specific latent (z^i).
        quantiles:           iterable of quantile levels tau_k.
        gate_factory:        optional callable ``(task, num_quantiles) ->
                             PhysicsGate``; defaults to the Wind/PV/Load
                             physical rules shipped with the repo.
    """

    TASK_NAMES = ("load", "wind", "pv")

    def __init__(
        self,
        look_back: int,
        horizon: int,
        n_targets: int,
        common_feat_dim: int,
        future_feat_dims: dict[str, int],
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        d_fut: int = 128,
        quantiles: Iterable[float] = DEFAULT_QUANTILES,
        gate_factory: callable | None = None,
    ) -> None:
        super().__init__()
        self.look_back = look_back
        self.horizon = horizon
        self.n_targets = n_targets
        self.quantiles = torch.tensor(list(quantiles), dtype=torch.float32)

        # ---- (1) Shared context encoder ----
        self.encoder = SharedContextEncoder(
            input_dim=n_targets + common_feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
        )
        d_hist = look_back * d_model     # dim of flattened h_global

        # ---- (2) Cross-task dependency-aware alignment ----
        self.alignment = CrossTaskAlignment(
            horizon=horizon,
            feat_dims=future_feat_dims,
            d_fut=d_fut,
            d_hist=d_hist,
        )

        # ---- (3) Physics-constrained quantile decoder ----
        self.decoder = MultiTaskQuantileDecoder(
            latent_dim=d_fut,
            horizon=horizon,
            quantiles=self.quantiles,
            gate_factory=gate_factory,
        )

        # ---- (4) Uncertainty-aware multi-task loss ----
        self.criterion = UncertaintyAwareMultiTaskLoss(
            num_tasks=len(self.TASK_NAMES),
            quantiles=self.quantiles,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x_hist: torch.Tensor,                              # (B, L, N + D_c)
        fut_feats: dict[str, torch.Tensor],                # per-task future feats
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x_hist:     concatenated historical target trajectories
                        and common exogenous features.
            fut_feats:  dict  {"load": (B, H, D_load), ...}.
        Returns:
            dict {"load": (B, H, K), "wind": ..., "pv": ...} of bounded
            quantile forecasts.
        """
        # (1) Global historical context
        h_global = self.encoder(x_hist)                    # (B, L * d_model)

        # (2) Cross-task alignment  -> task latents
        latents = self.alignment(h_global, fut_feats)      # {task: (B, d_fut)}

        # (3) Physics-constrained quantile decoding
        return self.decoder(latents, fut_feats)            # {task: (B, H, K)}

    # ------------------------------------------------------------------
    # Training loss convenience
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        preds: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Wraps ``self.criterion`` so the model can be called end-to-end."""
        return self.criterion(preds, target)
