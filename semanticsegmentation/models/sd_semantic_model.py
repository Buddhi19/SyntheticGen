import os
import sys
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(main_dir)

from UrbanMamba.classification.models.vmamba import LayerNorm2d  # noqa: E402
from UrbanMamba.semanticsegmentation.models.Mamba_backbone import Backbone_VSSM  # noqa: E402
from UrbanMamba.semanticsegmentation.models.SemanticDecoder import SemanticDecoder  # noqa: E402
from UrbanMamba.semanticsegmentation.models.sd_feature_extractor import SDFrozenFeatureExtractor  # noqa: E402
from UrbanMamba.semanticsegmentation.models.sd_fusion import SDTimestepAggregator  # noqa: E402


class SemanticMambaWithSDFP(nn.Module):
    """
    Example single-image semantic segmentation model:
      VMamba encoder + SemanticDecoder, optionally fused with frozen SD multi-scale/multi-timestep features.

    Notes:
      - Stable Diffusion modules are frozen; only small adapters/gates + decoder/head train by default.
      - The SD extractor expects images in [0,1] by default (set sd_assume_0_1=False if already in [-1,1]).
    """

    def __init__(
        self,
        *,
        num_classes: int,
        pretrained: Optional[str] = None,
        use_sd_fusion: bool = False,
        sd_base_model: Optional[str] = None,
        sd_timesteps: Sequence[int] = (50, 150, 250),
        sd_prompt: str = "aerial view, satellite image",
        sd_proj_dim: int = 128,
        sd_dtype: torch.dtype = torch.float16,
        sd_device: Union[str, torch.device] = "cuda",
        sd_assume_0_1: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.sd_assume_0_1 = bool(sd_assume_0_1)
        self.use_sd_fusion = bool(use_sd_fusion and sd_base_model is not None)

        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        channel_first = self.encoder.channel_first

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        norm_layer = _NORMLAYERS.get(str(kwargs["norm_layer"]).lower(), None)
        ssm_act_layer = _ACTLAYERS.get(str(kwargs["ssm_act_layer"]).lower(), None)
        mlp_act_layer = _ACTLAYERS.get(str(kwargs["mlp_act_layer"]).lower(), None)

        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ["norm_layer", "ssm_act_layer", "mlp_act_layer"]}

        self.decoder = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            use_sd_fusion=self.use_sd_fusion,
            **clean_kwargs,
        )

        self.seg_head = nn.Conv2d(in_channels=128, out_channels=int(num_classes), kernel_size=1, bias=True)

        if self.use_sd_fusion:
            self.sd_extractor = SDFrozenFeatureExtractor(
                str(sd_base_model),
                timesteps=sd_timesteps,
                prompt=sd_prompt,
                dtype=sd_dtype,
                device=sd_device,
            )
            self.sd_agg = SDTimestepAggregator(proj_dim=int(sd_proj_dim))
        else:
            self.sd_extractor = None
            self.sd_agg = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)

        sd_feats = None
        if self.use_sd_fusion:
            feats_per_t = self.sd_extractor(x, assume_0_1=self.sd_assume_0_1)
            sd_feats = self.sd_agg(feats_per_t)

        dec = self.decoder(feats, sd_feats=sd_feats)
        logits = self.seg_head(dec)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

