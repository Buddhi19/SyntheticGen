#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate synthetic (image, single-class mask) pairs with a 5-channel joint diffusion model."""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version

from .dataset_loveda import load_class_names


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0")

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic image+mask pairs from a joint diffusion model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained checkpoint directory.")
    unet_checkpoint_group = parser.add_mutually_exclusive_group()
    unet_checkpoint_group.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="If set, load UNet weights from checkpoint-{step} under --checkpoint.",
    )
    unet_checkpoint_group.add_argument(
        "--unet_checkpoint",
        type=str,
        default=None,
        help="Optional path to an Accelerate checkpoint directory containing an 'unet' subfolder (e.g. outputs/sdseg/checkpoint-1000).",
    )
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save generated samples.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="generic",
        choices=["loveda", "generic"],
        help="Dataset type to infer class names if class_names_json is not provided.",
    )
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes.")
    parser.add_argument("--class_names_json", type=str, default=None, help="Optional class names JSON file.")
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="Prompt template for class conditioning (defaults to training_config.json if present).",
    )
    parser.add_argument("--image_size", type=int, default=512, help="Generated image size (square).")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "ddpm"],
        help="Scheduler to use for sampling.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base seed; each sample uses seed+idx.")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ema", action="store_true", help="Load EMA UNet if available.")
    parser.add_argument(
        "--class_id",
        type=int,
        default=None,
        help="If set, generate all samples for this class id; otherwise sample uniformly over classes.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.0,
        help="Threshold applied to the generated mask latent after upsampling.",
    )
    return parser.parse_args()


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def _load_prompt_template(checkpoint: Path) -> Optional[str]:
    config_path = checkpoint / "training_config.json"
    if not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = data.get("prompt_template")
    return str(value) if value else None


def _vae_decode(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latents)
    if isinstance(decoded, torch.Tensor):
        return decoded
    if hasattr(decoded, "sample"):
        return decoded.sample
    if isinstance(decoded, (tuple, list)) and decoded:
        return decoded[0]
    raise TypeError(f"Unexpected VAE decode output type: {type(decoded)}")


def _save_uint8_rgb(image: torch.Tensor, save_path: Path) -> None:
    image = image.detach().cpu()
    image = (image / 2.0 + 0.5).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(save_path)


def _save_uint8_mask(mask: torch.Tensor, save_path: Path) -> None:
    mask = mask.detach().cpu()
    array = (mask.squeeze(0).squeeze(0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(save_path)


def _has_pipeline_components(checkpoint: Path) -> bool:
    return all((checkpoint / subfolder).is_dir() for subfolder in ("tokenizer", "text_encoder", "vae", "scheduler"))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    if args.image_size % 8 != 0:
        raise ValueError("--image_size must be divisible by 8 for Stable Diffusion latents.")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    unet_checkpoint = None
    if args.unet_checkpoint is not None:
        unet_checkpoint = Path(args.unet_checkpoint)
    elif args.checkpoint_step is not None:
        unet_checkpoint = checkpoint / f"checkpoint-{args.checkpoint_step}"
    else:
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
        if (
            match
            and not _has_pipeline_components(checkpoint)
            and _has_pipeline_components(checkpoint.parent)
        ):
            unet_checkpoint = checkpoint
            checkpoint = checkpoint.parent
        else:
            unet_checkpoint = checkpoint

    if not unet_checkpoint.exists():
        raise FileNotFoundError(f"UNet checkpoint not found: {unet_checkpoint}")

    if args.class_names_json is None:
        default_class_names = checkpoint / "class_names.json"
        if default_class_names.exists():
            args.class_names_json = str(default_class_names)

    class_names, num_classes = load_class_names(args.class_names_json, args.num_classes, args.dataset)

    if args.prompt_template is None:
        args.prompt_template = _load_prompt_template(checkpoint) or "a remote sensing image and mask of {class_name}"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    weight_dtype = resolve_dtype(args.dtype, device)

    unet_subfolder = "unet_ema" if args.use_ema else "unet"
    if not (unet_checkpoint / unet_subfolder).is_dir():
        raise FileNotFoundError(f"UNet subfolder '{unet_subfolder}' not found in {unet_checkpoint}")

    if unet_checkpoint != checkpoint:
        logger.info(f"Loading UNet from {unet_checkpoint}/{unet_subfolder} (rest from {checkpoint})")

    tokenizer = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(checkpoint, subfolder="text_encoder", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(unet_checkpoint, subfolder=unet_subfolder, torch_dtype=weight_dtype)

    if unet.config.in_channels != 5 or unet.config.out_channels != 5:
        raise ValueError(
            "This generator requires a 5-channel joint model (in_channels=5, out_channels=5). "
            f"Got in_channels={unet.config.in_channels}, out_channels={unet.config.out_channels}."
        )

    if args.scheduler == "ddim":
        scheduler = DDIMScheduler.from_pretrained(checkpoint, subfolder="scheduler")
    else:
        scheduler = DDPMScheduler.from_pretrained(checkpoint, subfolder="scheduler")

    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    text_encoder.eval()
    vae.eval()
    unet.eval()

    latent_h = args.image_size // 8
    latent_w = args.image_size // 8

    os.makedirs(args.save_dir, exist_ok=True)
    images_dir = Path(args.save_dir) / "images"
    masks_dir = Path(args.save_dir) / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for sample_idx in tqdm(range(args.num_samples), desc="Samples"):
        sample_seed = None
        if args.seed is not None:
            sample_seed = args.seed + sample_idx
        generator = None
        if sample_seed is not None:
            generator = torch.Generator(device=device).manual_seed(sample_seed)

        if args.class_id is not None:
            class_id = int(args.class_id)
        else:
            class_id = int(torch.randint(0, num_classes, (1,), generator=generator, device=device).item())
        if not (0 <= class_id < num_classes):
            raise ValueError(f"class_id must be in [0, {num_classes - 1}], got {class_id}")

        class_name = class_names[class_id]
        prompt = args.prompt_template.format(class_name=class_name, class_id=class_id)
        text_inputs = tokenizer(
            [prompt],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]

        latents = torch.randn(
            (1, 5, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=weight_dtype,
        )
        latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(args.num_inference_steps, device=device)
        with torch.no_grad():
            for timestep in scheduler.timesteps:
                noise_pred = unet(latents, timestep, encoder_hidden_states=prompt_embeds).sample
                latents = scheduler.step(noise_pred, timestep, latents).prev_sample

        image_latents = latents[:, :4, :, :]
        mask_latents = latents[:, 4:5, :, :]

        image = _vae_decode(vae, image_latents / vae.config.scaling_factor)[0]
        mask = F.interpolate(mask_latents.float(), size=(args.image_size, args.image_size), mode="nearest")
        mask = (mask > args.mask_threshold).to(dtype=torch.float32)

        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(class_name).lower())
        stem = f"sample_{sample_idx:05d}_class{class_id:02d}_{safe_name}"
        image_path = images_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}_mask.png"
        _save_uint8_rgb(image, image_path)
        _save_uint8_mask(mask, mask_path)

        metadata.append(
            {
                "index": sample_idx,
                "seed": sample_seed,
                "class_id": class_id,
                "class_name": class_name,
                "prompt": prompt,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )

    (Path(args.save_dir) / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(metadata)} samples under {args.save_dir}")


if __name__ == "__main__":
    main()
