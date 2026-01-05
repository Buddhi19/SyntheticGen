import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(main_dir)

from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel  # noqa: E402


@dataclass(frozen=True)
class SDFeatureSpec:
    name: str
    up_block_index: int


class SDFrozenFeatureExtractor(nn.Module):
    """
    Frozen Stable Diffusion feature extractor.

    Workflow:
      1) Encode an RGB image into SD latents with the VAE.
      2) Add noise at a small set of diffusion timesteps.
      3) Run the SD UNet (frozen) and capture intermediate activations from chosen up_blocks.
      4) Return per-timestep features (caller aggregates/fuses).
    """

    def __init__(
        self,
        base_model: str,
        *,
        timesteps: Sequence[int] = (50, 150, 250),
        prompt: str = "aerial view, satellite image",
        dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
        feature_specs: Sequence[SDFeatureSpec] = (
            SDFeatureSpec("d32", 1),
            SDFeatureSpec("d16", 2),
            SDFeatureSpec("d8", 3),
        ),
    ) -> None:
        super().__init__()

        self.base_model = base_model
        self.timesteps = [int(t) for t in timesteps]
        self.prompt = str(prompt)
        self.dtype = dtype
        self.device = torch.device(device)
        self.feature_specs = tuple(feature_specs)

        self.tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=dtype)
        self.vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype)
        self.unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=dtype)

        # Use DDPM scheduler for consistent add_noise behavior (even if base config is PNDM/DDIM).
        self.scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()

        for module in (self.text_encoder, self.vae, self.unet):
            for param in module.parameters():
                param.requires_grad_(False)

        self._feat_store: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

        self._prompt_embeds_cache: Optional[torch.Tensor] = None

        self.to(self.device)

    def _register_hooks(self) -> None:
        def make_hook(name: str):
            def hook(_module, _inputs, output):
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if not isinstance(output, torch.Tensor):
                    return
                self._feat_store[name] = output

            return hook

        for spec in self.feature_specs:
            if spec.up_block_index < 0 or spec.up_block_index >= len(self.unet.up_blocks):
                raise ValueError(
                    f"Invalid up_block_index={spec.up_block_index} for UNet with {len(self.unet.up_blocks)} up_blocks."
                )
            self._hooks.append(self.unet.up_blocks[spec.up_block_index].register_forward_hook(make_hook(spec.name)))

    @torch.no_grad()
    def _encode_prompt(self, batch_size: int) -> torch.Tensor:
        # Cache single-prompt embeds and repeat to batch size.
        if self._prompt_embeds_cache is None:
            tokens = self.tokenizer(
                [self.prompt],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device=self.device)
            embeds = self.text_encoder(input_ids)[0]  # (1, T, C)
            self._prompt_embeds_cache = embeds.to(dtype=self.dtype, device=self.device)

        return self._prompt_embeds_cache.expand(batch_size, -1, -1).contiguous()

    @torch.no_grad()
    def forward(self, images: torch.Tensor, *, assume_0_1: bool = True) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            images: (B,3,H,W) float tensor. If `assume_0_1=True`, values are expected in [0,1].
                    Otherwise, values are expected in [-1,1] already.
            assume_0_1: Whether to map inputs from [0,1] -> [-1,1] for the VAE.

        Returns:
            List[Dict[str,Tensor]] with length T (timesteps). Each dict maps feature name -> tensor.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images as (B,3,H,W), got {tuple(images.shape)}")

        batch_size = int(images.shape[0])
        x = images.to(device=self.device, dtype=self.dtype)
        if assume_0_1:
            x = x * 2.0 - 1.0

        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        text_embeds = self._encode_prompt(batch_size)

        feats_per_t: List[Dict[str, torch.Tensor]] = []
        for t in self.timesteps:
            self._feat_store = {}
            t_tensor = torch.full((batch_size,), int(t), device=self.device, dtype=torch.long)
            noise = torch.randn_like(latents)
            noised = self.scheduler.add_noise(latents, noise, t_tensor)

            _ = self.unet(noised, t_tensor, encoder_hidden_states=text_embeds)

            missing = [spec.name for spec in self.feature_specs if spec.name not in self._feat_store]
            if missing:
                raise RuntimeError(f"Missing hooked features after UNet forward: {missing}")

            feats_per_t.append({name: self._feat_store[name] for name in (spec.name for spec in self.feature_specs)})

        return feats_per_t

    def extra_repr(self) -> str:
        return (
            f"base_model={self.base_model!r}, timesteps={self.timesteps}, "
            f"prompt={self.prompt!r}, dtype={str(self.dtype).replace('torch.', '')}, device={self.device}"
        )

