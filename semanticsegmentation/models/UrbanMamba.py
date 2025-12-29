import os
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(main_dir)

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from UrbanMamba.semanticsegmentation.models.Mamba_backbone import Backbone_VSSM
from UrbanMamba.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from UrbanMamba.semanticsegmentation.models.ChangeDecoder import ChangeDecoder
from UrbanMamba.semanticsegmentation.models.SemanticDecoder import SemanticDecoder

class ChangeMambaSCD(nn.Module):
    def __init__(self, output_cd, output_clf, pretrained,  **kwargs):
        super(ChangeMambaSCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
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

        self.channel_first = self.encoder.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)


        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}

        self.decoder = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)


    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)

        # Decoder processing - passing encoder outputs to the decoder
        output = self.decoder(pre_features)


        output = self.aux_clf(output_T1)
        output = F.interpolate(output_T1, size=pre_data.size()[-2:], mode='bilinear')

        return output
