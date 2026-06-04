"""
Physics-Constrained Quantile Decoder  (Sec. 3.3 of the DAFQ-Net paper)

For every task i in {Load, Wind, PV} the decoder

    1. linearly projects the task-specific latent vector z^i into an
       unconstrained quantile estimate  V^i  of shape (B, H, K);
    2. multiplies it element-wise with a physics gate G^i  of the same
       shape, so that selected operational bounds (cut-in/cut-out,
       solar elevation, non-negativity) are enforced structurally;
    3. applies a ReLU to ensure non-negativity.

The choice of G^i depends on the energy modality.  See ``physics_gates.py``
for the three concrete implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from physics_gates import LoadGate, PhysicsGate, PVGate, WindGate


# ----------------------------------------------------------------------
# Single-task head
# ----------------------------------------------------------------------
class QuantileHead(nn.Module):
    """
    Linear projection  +  physics gate  +  ReLU.

    Args:
        latent_dim:   dim of the input task-specific latent z^i.
        horizon:      forecasting horizon H.
        num_quantiles: number of quantile levels K.
        gate:         a ``PhysicsGate`` instance whose forward returns
                      a tensor of shape (B, H, K).
    """

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        num_quantiles: int,
        gate: PhysicsGate,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.num_quantiles = num_quantiles
        self.gate = gate

        # Linear projection to (H * K) raw quantile estimates.
        self.proj = nn.Linear(latent_dim, horizon * num_quantiles)

    def forward(
        self,
        z_task: torch.Tensor,            # (B, latent_dim)
        horizon_features: torch.Tensor,  # (B, H, *)  task-specific future features
    ) -> torch.Tensor:
        B = z_task.size(0)
        v = self.proj(z_task)                                     # (B, H*K)
        v = v.view(B, self.horizon, self.num_quantiles)           # (B, H, K)
        g = self.gate(z_task, horizon_features)                   # (B, H, K)
        return torch.relu(v * g)                                  # (B, H, K)


# ----------------------------------------------------------------------
# Multi-task decoder
# ----------------------------------------------------------------------
class MultiTaskQuantileDecoder(nn.Module):
    """
    Wraps three ``QuantileHead``s (Load, Wind, PV) into a single module.

    Args:
        latent_dim:    dim of every task's latent z^i.
        horizon:       forecasting horizon H.
        quantiles:     1-D iterable of quantile levels (used to size the
                       broadcast inside the gates).
        gate_factory:  callable ``gate_factory(task_name, num_quantiles)``
                       returning a ``PhysicsGate``.  If ``None`` we use
                       the default physical rules: Wind, PV, Load.
    """

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        quantiles: list[float] | torch.Tensor,
        gate_factory: callable | None = None,
    ) -> None:
        super().__init__()
        if isinstance(quantiles, torch.Tensor):
            num_quantiles = quantiles.numel()
        else:
            num_quantiles = len(quantiles)

        if gate_factory is None:
            gate_factory = self._default_gate_factory

        self.heads = nn.ModuleDict({
            "load": QuantileHead(latent_dim, horizon, num_quantiles,
                                 gate_factory("load", num_quantiles)),
            "wind": QuantileHead(latent_dim, horizon, num_quantiles,
                                 gate_factory("wind", num_quantiles)),
            "pv":   QuantileHead(latent_dim, horizon, num_quantiles,
                                 gate_factory("pv",   num_quantiles)),
        })

    @staticmethod
    def _default_gate_factory(task: str, num_quantiles: int) -> PhysicsGate:
        if task == "wind":
            return WindGate(num_quantiles=num_quantiles)
        if task == "pv":
            return PVGate(num_quantiles=num_quantiles)
        return LoadGate(num_quantiles=num_quantiles)

    def forward(
        self,
        latents: dict[str, torch.Tensor],          # {"load": (B,d), "wind":..., "pv":...}
        horizon_features: dict[str, torch.Tensor],  # {"load": (B,H,*), ...}
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict  {"load": (B,H,K), "wind": ..., "pv": ...} of bounded
            quantile forecasts.
        """
        out: dict[str, torch.Tensor] = {}
        for name in ("load", "wind", "pv"):
            out[name] = self.heads[name](latents[name], horizon_features[name])
        return out
