from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


def infer_time_embed_dim_from_config(block_out_channels) -> int:
    if not block_out_channels:
        raise ValueError("block_out_channels must be a non-empty sequence.")
    return int(block_out_channels[0]) * 4


class RatioProjector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        hidden_mult: int = 4,
        *,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.input_dim = int(input_dim) if input_dim is not None else self.num_classes
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else max(1, int(hidden_mult) * self.num_classes)

        layers = [nn.Linear(self.input_dim, self.hidden_dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Linear(self.hidden_dim, self.embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        ratios: torch.Tensor,
        known_mask: Optional[torch.Tensor] = None,
        *,
        known_sum: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if known_mask is None:
            return self.net(ratios)

        if ratios.shape[-1] != self.num_classes or known_mask.shape[-1] != self.num_classes:
            raise ValueError(
                f"Expected ratios/mask last dim == {self.num_classes}, "
                f"got ratios={tuple(ratios.shape)} mask={tuple(known_mask.shape)}."
            )

        include_known_sum = self.input_dim == (2 * self.num_classes + 1)
        if include_known_sum and known_sum is None:
            known_sum = (ratios * known_mask).sum(dim=-1, keepdim=True)
        feat = torch.cat([ratios, known_mask], dim=-1)
        if include_known_sum:
            feat = torch.cat([feat, known_sum], dim=-1)
        return self.net(feat)


def build_ratio_projector_from_state_dict(
    state_dict: Dict[str, torch.Tensor], num_classes: int, embed_dim: int
) -> RatioProjector:
    """Instantiate a RatioProjector that matches a saved state dict (supports old/new input dims)."""
    w0 = state_dict.get("net.0.weight")
    if w0 is None or not hasattr(w0, "shape") or len(w0.shape) != 2:
        raise ValueError("Invalid ratio projector state dict: missing net.0.weight.")
    hidden_dim = int(w0.shape[0])
    input_dim = int(w0.shape[1])
    layer_norm = "net.1.weight" in state_dict and state_dict["net.1.weight"].numel() == hidden_dim
    return RatioProjector(
        num_classes=num_classes,
        embed_dim=embed_dim,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        layer_norm=layer_norm,
    )


class ResidualFiLMGate(nn.Module):
    def __init__(self, ratio_embed_dim: int, n_down_blocks: int) -> None:
        super().__init__()
        self.n_blocks = n_down_blocks + 1
        self.proj = nn.Linear(ratio_embed_dim, 2 * self.n_blocks)

    def forward(self, ratio_emb, down_residuals, mid_residual):
        params = self.proj(ratio_emb)
        gamma, beta = params.chunk(2, dim=1)
        out_down = []
        for idx, residual in enumerate(down_residuals):
            g = gamma[:, idx].view(-1, 1, 1, 1).to(residual.dtype)
            b = beta[:, idx].view(-1, 1, 1, 1).to(residual.dtype)
            out_down.append(residual * (1.0 + g) + b)
        g = gamma[:, -1].view(-1, 1, 1, 1).to(mid_residual.dtype)
        b = beta[:, -1].view(-1, 1, 1, 1).to(mid_residual.dtype)
        out_mid = mid_residual * (1.0 + g) + b
        return tuple(out_down), out_mid


class PerChannelResidualFiLMGate(nn.Module):
    def __init__(
        self,
        ratio_embed_dim: int,
        down_channels: List[int],
        mid_channels: int,
        init_zero: bool = True,
    ) -> None:
        super().__init__()
        self.down_mlps = nn.ModuleList([nn.Linear(ratio_embed_dim, 2 * int(c)) for c in down_channels])
        self.mid_mlp = nn.Linear(ratio_embed_dim, 2 * int(mid_channels))

        if init_zero:
            for mlp in self.down_mlps:
                nn.init.zeros_(mlp.weight)
                nn.init.zeros_(mlp.bias)
            nn.init.zeros_(self.mid_mlp.weight)
            nn.init.zeros_(self.mid_mlp.bias)

    def forward(
        self,
        ratio_emb: torch.Tensor,
        down_residuals: Tuple[torch.Tensor, ...],
        mid_residual: torch.Tensor,
    ):
        if len(down_residuals) != len(self.down_mlps):
            raise ValueError(f"Expected {len(self.down_mlps)} down residuals, got {len(down_residuals)}.")

        out_down = []
        for residual, mlp in zip(down_residuals, self.down_mlps):
            params = mlp(ratio_emb).to(dtype=residual.dtype)
            params = params.unsqueeze(-1).unsqueeze(-1)
            gamma, beta = params.chunk(2, dim=1)
            out_down.append(residual * (1.0 + gamma) + beta)

        mid_params = self.mid_mlp(ratio_emb).to(dtype=mid_residual.dtype).unsqueeze(-1).unsqueeze(-1)
        mid_gamma, mid_beta = mid_params.chunk(2, dim=1)
        out_mid = mid_residual * (1.0 + mid_gamma) + mid_beta
        return tuple(out_down), out_mid
