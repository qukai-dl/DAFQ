"""
Multi-Task Uncertainty-Aware Loss for DAFQ-Net.

Implements:
    (1) Pinball (Quantile) loss for probabilistic forecasting.
    (2) Homoscedastic uncertainty-aware global loss that adaptively
        balances the three energy tasks (Load, Wind, PV) via learnable
        log-variance parameters  s_i = log(sigma_i^2).

Reference: Section 3.4 ("Multi-Task Uncertainty Optimization") of the
DAFQ-Net paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Pinball / Quantile loss
# ----------------------------------------------------------------------
def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """
    Element-wise pinball loss.

    Args:
        y_true:  (B, H)         ground-truth target over the horizon.
        y_pred:  (B, H, K)      predicted quantiles (K = #quantile levels).
        taus:    (K,)           quantile levels in (0, 1).

    Returns:
        (B, H, K) pinball loss, not reduced.
    """
    # broadcast quantile levels over (B, H)
    tau = taus.view(1, 1, -1)
    y_true = y_true.unsqueeze(-1)                       # (B, H, 1)
    error = y_true - y_pred                              # (B, H, K)
    return torch.maximum(tau * error, (tau - 1.0) * error)


# ----------------------------------------------------------------------
# Homoscedastic uncertainty-aware multi-task loss
# ----------------------------------------------------------------------
class UncertaintyAwareMultiTaskLoss(nn.Module):
    """
    Implements the global objective:

        J(Theta, s) = sum_i [ exp(-s_i) * L_i + s_i ]

    where s_i = log(sigma_i^2) are learnable log-variance parameters and
    L_i is the average pinball loss of task i over (H, K).

    Args:
        num_tasks: number of energy tasks (default 3: Load, Wind, PV).
        quantiles: 1-D tensor of quantile levels, e.g.
                   [0.05, 0.10, ..., 0.95].
        init_log_vars: optional initial values for s_i.  If None, init=0.
    """

    def __init__(
        self,
        num_tasks: int = 3,
        quantiles: torch.Tensor | list[float] | None = None,
        init_log_vars: list[float] | None = None,
    ) -> None:
        super().__init__()
        if quantiles is None:
            quantiles = torch.linspace(0.05, 0.95, 19)
        elif isinstance(quantiles, list):
            quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.register_buffer("quantiles", quantiles.float())

        if init_log_vars is None:
            init_log_vars = [0.0] * num_tasks
        # s_i = log(sigma_i^2) — learnable parameters
        self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))

    # ---- task-specific loss ----
    def task_loss(
        self,
        y_true: torch.Tensor,        # (B, H)
        y_pred: torch.Tensor,        # (B, H, K)
        task_index: int,
    ) -> torch.Tensor:
        """Average pinball loss of one task over batch/horizon/quantile."""
        l = pinball_loss(y_true, y_pred, self.quantiles.to(y_pred.device))
        return l.mean()

    # ---- global objective ----
    def forward(
        self,
        preds: dict[str, torch.Tensor],   # {"load": (B,H,K), "wind": ..., "pv": ...}
        target: dict[str, torch.Tensor],  # {"load": (B,H),   "wind": ..., "pv": ...}
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns:
            total_loss : scalar tensor, the global objective J.
            info       : dict with per-task loss, weights exp(-s_i),
                         and regularisation terms (for logging).
        """
        device = self.log_vars.device
        total = torch.zeros((), device=device)
        info: dict[str, torch.Tensor] = {}

        for i, name in enumerate(["load", "wind", "pv"]):
            l_i = self.task_loss(target[name], preds[name], i)
            w_i = torch.exp(-self.log_vars[i])
            reg = self.log_vars[i]
            total = total + w_i * l_i + reg

            info[f"{name}_loss"] = l_i.detach()
            info[f"{name}_weight"] = w_i.detach()
            info[f"{name}_log_var"] = reg.detach()

        return total, info
