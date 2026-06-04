"""
Shared Context Encoder  (Sec. 3.1 of the DAFQ-Net paper)

The shared encoder transforms the multivariate historical context
(concat of historical target trajectories Y_{t-L+1:t} in R^{L x N} and
common exogenous features X^com_{t-L+1:t} in R^{L x D_c}) into a single
global context vector  h_global in R^{L * d_model}.

Steps:

    (a) Feature concatenation + linear embedding  to d_model.
    (b) Sinusoidal positional encoding added to the embedding.
    (c) Stacked Transformer encoder layers (MHSA + FFN).
    (d) Flatten  ->  h_global.

This module is shared across the three energy tasks (Load, Wind, PV) so
that the latent representation captures *cross-variable* temporal
correlations of the entire microgrid system.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Sinusoidal positional encoding
# ----------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Input  : (B, L, d_model)
    Output : (B, L, d_model)   (added to the input by the caller).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))                   # (1, max_len, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ----------------------------------------------------------------------
# Shared context encoder
# ----------------------------------------------------------------------
class SharedContextEncoder(nn.Module):
    """
    Transformer-based shared context encoder.

    Args:
        input_dim:   N + D_c  (number of targets + dimension of common
                               exogenous features).
        d_model:     internal model dimension.
        nhead:       number of attention heads in each MHSA layer.
        num_layers:  number of stacked Transformer encoder layers.
        dim_ff:      hidden size of the FFN sub-layer (defaults to 4*d_model).
        dropout:     dropout rate used in positional encoding, MHSA, FFN.
        max_len:     maximum sequence length for the sinusoidal table.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int | None = None,
        dropout: float = 0.2,
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        dim_ff = dim_ff or 4 * d_model

        # (a) Linear embedding of the concatenated features.
        self.embed = nn.Linear(input_dim, d_model)

        # (b) Sinusoidal positional encoding.
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # (c) Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.d_model = d_model

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_hist: (B, L, N + D_c)  historical context, already
                    concatenated feature-wise (Y || X^com).
        Returns:
            h_global: (B, L * d_model) flattened global context.
        """
        # (a) embed to d_model, (b) add positional encoding
        h = self.embed(x_hist)                     # (B, L, d_model)
        h = self.pos_enc(h)                        # (B, L, d_model)

        # (c) stacked Transformer encoder
        h = self.transformer(h)                     # (B, L, d_model)

        # (d) flatten into a single global context vector
        return h.flatten(start_dim=1)              # (B, L * d_model)
