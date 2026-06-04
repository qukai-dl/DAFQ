"""
Physics-Constrained Gating Tensors for the Quantile Decoder.

Each energy modality has its own physical operating bounds.  Instead of
penalising violations through the loss, DAFQ-Net hard-codes selected
bounds *structurally* into a Hadamard gate that multiplies the unconstrained
quantile estimates before they leave the decoder.

Implemented rules (Sec. 3.3 of the paper):

    1) Wind  : G^Wind_{h,k} = I(v_in <= v_{t+h} <= v_out) * sigmoid(MLP(z^Wind))
    2) PV     : piecewise smooth function of solar elevation angle
                (sin-based transition zone to remove the discontinuity jump
                 at sunrise / sunset).
    3) Load   : unity gate  G^Load = 1  (ReLU already enforces non-negativity).

The output of every gate has shape (B, H, K) so it can be broadcast over
the K quantile levels.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------
class PhysicsGate(nn.Module):
    """Abstract physics gate.  Sub-classes must implement ``forward``."""

    def forward(
        self,
        z_task: torch.Tensor,            # (B, d_fut)  task latent vector
        horizon_features: torch.Tensor,  # (B, H, *)   task-specific future features
    ) -> torch.Tensor:                  # (B, H, K)
        raise NotImplementedError


# ----------------------------------------------------------------------
# Wind gate — cut-in / cut-out safety + learnable reliability factor
# ----------------------------------------------------------------------
class WindGate(PhysicsGate):
    """
    Implements

        G^Wind_{h,k} = I(v_in <= v_{t+h} <= v_out) * sigmoid(MLP(z^Wind))

    The ``MLP`` produces a per-quile reliability factor.  For simplicity
    we let the MLP output ``K`` independent values (one per quantile level).
    The indicator is broadcast over the K dimension.

    Args:
        v_in, v_out:       cut-in / cut-out wind speeds (m/s).
        wind_speed_index:  index of wind speed in ``horizon_features`` dim.
        hidden:            hidden size of the reliability MLP.
        num_quantiles:     number of quantile levels K.
    """

    def __init__(
        self,
        v_in: float = 3.0,
        v_out: float = 25.0,
        wind_speed_index: int = 0,
        hidden: int = 32,
        num_quantiles: int = 19,
    ) -> None:
        super().__init__()
        self.v_in = v_in
        self.v_out = v_out
        self.wind_speed_index = wind_speed_index
        self.num_quantiles = num_quantiles

        self.reliability_mlp = nn.Sequential(
            nn.Linear(1, hidden),       # input is the broadcast z norm
            nn.ReLU(),
            nn.Linear(hidden, num_quantiles),
        )

    def forward(
        self,
        z_task: torch.Tensor,            # (B, d_fut)
        horizon_features: torch.Tensor,  # (B, H, D_wind)
    ) -> torch.Tensor:
        B, H, _ = horizon_features.shape
        v = horizon_features[..., self.wind_speed_index]               # (B, H)
        # Hard safety switch — cut-in / cut-out indicator
        safety = ((v >= self.v_in) & (v <= self.v_out)).float()        # (B, H)
        # Learnable reliability factor (per quantile)
        # Use a single scalar summary of z_task to feed the MLP so that
        # the output only depends on the latent state, not on the horizon.
        z_summary = z_task.mean(dim=-1, keepdim=True)                  # (B, 1)
        reliability = torch.sigmoid(
            self.reliability_mlp(z_summary)                            # (B, K)
        )                                                               # (B, K)
        # Broadcast: (B, H, 1) * (B, 1, K) -> (B, H, K)
        return safety.unsqueeze(-1) * reliability.unsqueeze(1)


# ----------------------------------------------------------------------
# PV gate — smooth elevation-based modulation
# ----------------------------------------------------------------------
class PVGate(PhysicsGate):
    """
    Implements the smooth elevation-based gate

        G^PV_{h,k} = 0                       if alpha < 0
                   = sin(alpha)              if 0 <= alpha <= eps
                   = 1                       if alpha > eps

    where ``alpha`` is the solar elevation angle (radians) and ``eps`` is
    the transition-zone threshold (e.g. 5 degrees).  The same gate is
    broadcast over the K quantile levels.

    Args:
        eps_deg: transition zone size in degrees.
        elevation_index: index of the elevation angle in
                         ``horizon_features``.  If the input feature is
                         already an angle in degrees, set
                         ``elevation_in_degrees=True``.
        elevation_in_degrees: whether the input feature is in degrees
                              (we convert to radians internally).
        num_quantiles: number of quantile levels K (used only for the
                       broadcast shape).
    """

    def __init__(
        self,
        eps_deg: float = 5.0,
        elevation_index: int = 0,
        elevation_in_degrees: bool = True,
        num_quantiles: int = 19,
    ) -> None:
        super().__init__()
        self.eps = math.radians(eps_deg)
        self.elevation_index = elevation_index
        self.elevation_in_degrees = elevation_in_degrees
        self.num_quantiles = num_quantiles

    def forward(
        self,
        z_task: torch.Tensor,            # (B, d_fut) — unused
        horizon_features: torch.Tensor,  # (B, H, D_pv)
    ) -> torch.Tensor:
        alpha = horizon_features[..., self.elevation_index]
        if self.elevation_in_degrees:
            alpha = torch.deg2rad(alpha)

        gate = torch.where(
            alpha < 0.0,
            torch.zeros_like(alpha),
            torch.where(
                alpha <= self.eps,
                torch.sin(alpha.clamp(min=0.0)),
                torch.ones_like(alpha),
            ),
        )                                                  # (B, H)
        return gate.unsqueeze(-1).expand(-1, -1, self.num_quantiles)


# ----------------------------------------------------------------------
# Load gate — unity (ReLU in the decoder already enforces non-negativity)
# ----------------------------------------------------------------------
class LoadGate(PhysicsGate):
    """G^Load = 1."""

    def __init__(self, num_quantiles: int = 19) -> None:
        super().__init__()
        self.num_quantiles = num_quantiles

    def forward(
        self,
        z_task: torch.Tensor,            # (B, d_fut) — unused
        horizon_features: torch.Tensor,  # (B, H, D_load)
    ) -> torch.Tensor:
        B, H, _ = horizon_features.shape
        return torch.ones(B, H, self.num_quantiles, device=z_task.device)
