import torch
from torch import nn
from typing import List, Tuple


def infer_time_embed_dim_from_config(block_out_channels) -> int:
    if not block_out_channels:
        raise ValueError("block_out_channels must be a non-empty sequence.")
    return int(block_out_channels[0]) * 4


class RatioProjector(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = max(1, hidden_mult * num_classes)
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, ratios: torch.Tensor) -> torch.Tensor:
        return self.net(ratios)


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
