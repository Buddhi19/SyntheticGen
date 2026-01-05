from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDProjector(nn.Module):
    """
    1x1 conv projection to a fixed width + lightweight normalization.

    Uses LazyConv2d so it can accept SD features from different base models.
    """

    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        self.proj = nn.LazyConv2d(out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels, eps=1e-6, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.norm(x)


class TimeAgg(nn.Module):
    """
    Attention-weighted aggregation over diffusion timesteps.

    weights: softmax(MLP(GAP(feat_t)))  -> per-sample weights over t
    out: sum_t weights_t * feat_t
    """

    def __init__(self, in_ch: int, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats_list: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feats_list) == 0:
            raise ValueError("feats_list must be non-empty")

        scores: List[torch.Tensor] = []
        for feat in feats_list:
            gap = feat.mean(dim=(2, 3))  # (B,C)
            scores.append(self.mlp(gap))  # (B,1)

        score_tensor = torch.stack(scores, dim=1)  # (B,T,1)
        weights = torch.softmax(score_tensor, dim=1)  # (B,T,1)

        out = torch.zeros_like(feats_list[0])
        for t, feat in enumerate(feats_list):
            out = out + weights[:, t].view(-1, 1, 1, 1) * feat
        return out


class SDTimestepAggregator(nn.Module):
    """
    Multi-scale SD feature projection + multi-timestep aggregation.

    Input: list of per-timestep dicts, e.g. [{d32:..., d16:..., d8:...}, ...]
    Output: dict with aggregated features per scale.
    """

    def __init__(
        self,
        *,
        proj_dim: int = 128,
        scales: Sequence[str] = ("d32", "d16", "d8"),
        time_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.scales = tuple(scales)
        self.projectors = nn.ModuleDict({s: SDProjector(out_channels=proj_dim) for s in self.scales})
        self.aggregators = nn.ModuleDict({s: TimeAgg(in_ch=proj_dim, hidden=time_hidden) for s in self.scales})

    def forward(self, feats_per_t: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(feats_per_t) == 0:
            raise ValueError("feats_per_t must be non-empty")

        out: Dict[str, torch.Tensor] = {}
        for scale in self.scales:
            projected: List[torch.Tensor] = []
            for per_t in feats_per_t:
                if scale not in per_t:
                    raise KeyError(f"Missing scale {scale!r} in per-timestep features: {list(per_t.keys())}")
                projected.append(self.projectors[scale](per_t[scale]))
            out[scale] = self.aggregators[scale](projected)
        return out


class SDFusionBlock(nn.Module):
    """
    Gated injection of SD features into a segmentation feature map.

    seg <- seg + sigmoid(gate(seg)) * adapter(sd)
    """

    def __init__(self, seg_channels: int = 128, out_channels: Optional[int] = None) -> None:
        super().__init__()
        out_channels = int(out_channels) if out_channels is not None else int(seg_channels)
        self.sd_adapter = nn.LazyConv2d(out_channels, kernel_size=1, bias=True)
        self.gate = nn.Sequential(nn.Conv2d(seg_channels, out_channels, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, seg_feat: torch.Tensor, sd_feat: torch.Tensor) -> torch.Tensor:
        sd_feat = F.interpolate(sd_feat, size=seg_feat.shape[-2:], mode="bilinear", align_corners=False)
        sd_feat = self.sd_adapter(sd_feat)
        gate = self.gate(seg_feat)
        return seg_feat + gate * sd_feat

