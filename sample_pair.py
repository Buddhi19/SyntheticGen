#!/usr/bin/env python
# coding=utf-8

"""Sample (layout, image) pairs using layout DDPM + ControlNet with ratio conditioning."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler, DDPMScheduler, UNet2DConditionModel, UNet2DModel

from ratio_conditioning import RatioProjector, ResidualFiLMGate, infer_time_embed_dim_from_config


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample layout and image pairs.")
    parser.add_argument("--layout_ckpt", type=str, required=True, help="Layout DDPM checkpoint directory.")
    parser.add_argument("--controlnet_ckpt", type=str, required=True, help="ControlNet checkpoint directory.")
    parser.add_argument("--base_model", type=str, default=None, help="Base SD model path or ID.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--ratios", type=str, default=None, help="Ratios as CSV or name:value pairs.")
    parser.add_argument("--ratios_json", type=str, default=None, help="JSON file with ratios list/dict.")
    parser.add_argument("--class_names_json", type=str, default=None, help="Optional class names JSON.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for image sampling.")
    parser.add_argument("--num_inference_steps_layout", type=int, default=50)
    parser.add_argument("--num_inference_steps_image", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scheduler", type=str, default="ddim", choices=["ddim", "ddpm"])
    return parser.parse_args()


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def _load_json(path: Path) -> Optional[Dict]:
    if not path or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_class_names(args, layout_ckpt: Path, controlnet_ckpt: Path, num_classes: int) -> List[str]:
    if args.class_names_json:
        data = _load_json(Path(args.class_names_json))
        if isinstance(data, list):
            return [str(x) for x in data]
    for candidate in [layout_ckpt / "class_names.json", controlnet_ckpt / "class_names.json"]:
        data = _load_json(candidate)
        if isinstance(data, list):
            return [str(x) for x in data]
    return [f"class_{i}" for i in range(num_classes)]


def _parse_ratios(ratios_str: Optional[str], ratios_json: Optional[str], num_classes: int, class_names: List[str]) -> torch.Tensor:
    if ratios_json:
        data = _load_json(Path(ratios_json))
        if isinstance(data, list):
            values = [float(x) for x in data]
        elif isinstance(data, dict):
            values = [0.0] * num_classes
            for key, value in data.items():
                idx = int(key) if str(key).isdigit() else class_names.index(key)
                values[idx] = float(value)
        else:
            raise ValueError("ratios_json must be a list or dict.")
    elif ratios_str:
        if ":" in ratios_str:
            values = [0.0] * num_classes
            for chunk in ratios_str.split(","):
                name, value = chunk.split(":")
                name = name.strip()
                idx = int(name) if name.isdigit() else class_names.index(name)
                values[idx] = float(value)
        else:
            values = [float(x) for x in ratios_str.split(",")]
    else:
        raise ValueError("Provide --ratios or --ratios_json.")

    if len(values) != num_classes:
        raise ValueError(f"Expected {num_classes} ratios, got {len(values)}.")
    ratios = torch.tensor(values, dtype=torch.float32)
    ratios = torch.clamp(ratios, min=0)
    total = ratios.sum()
    if total <= 0:
        raise ValueError("Ratios must sum to a positive value.")
    return ratios / total


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


def _save_label_map(label: torch.Tensor, save_path: Path) -> None:
    label = label.detach().cpu().numpy()
    if label.max() < 256:
        array = label.astype(np.uint8)
        mode = "L"
    else:
        array = label.astype(np.uint16)
        mode = "I;16"
    Image.fromarray(array, mode=mode).save(save_path)


def _load_ratio_projector(checkpoint_dir: Path, num_classes: int, embed_dim: int) -> RatioProjector:
    projector = RatioProjector(num_classes, embed_dim)
    state_path = checkpoint_dir / "ratio_projector.bin"
    if not state_path.is_file():
        raise FileNotFoundError(f"ratio_projector not found at {state_path}")
    projector.load_state_dict(torch.load(state_path, map_location="cpu"))
    return projector


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    weight_dtype = resolve_dtype(args.dtype, device)

    layout_ckpt = Path(args.layout_ckpt)
    controlnet_ckpt = Path(args.controlnet_ckpt)
    if not layout_ckpt.exists():
        raise FileNotFoundError(f"Layout checkpoint not found: {layout_ckpt}")
    if not controlnet_ckpt.exists():
        raise FileNotFoundError(f"ControlNet checkpoint not found: {controlnet_ckpt}")

    layout_unet = UNet2DModel.from_pretrained(layout_ckpt / "layout_unet")
    num_classes = layout_unet.config.in_channels
    layout_size = layout_unet.config.sample_size
    time_embed_dim_layout = infer_time_embed_dim_from_config(layout_unet.config.block_out_channels)
    layout_ratio_projector = _load_ratio_projector(layout_ckpt, num_classes, time_embed_dim_layout)
    layout_scheduler = DDPMScheduler.from_pretrained(layout_ckpt / "scheduler")

    class_names = _load_class_names(args, layout_ckpt, controlnet_ckpt, num_classes)
    ratios = _parse_ratios(args.ratios, args.ratios_json, num_classes, class_names)

    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    layout_unet.to(device, dtype=weight_dtype)
    layout_ratio_projector.to(device, dtype=weight_dtype)
    layout_unet.eval()
    layout_ratio_projector.eval()

    ratio_emb_layout = layout_ratio_projector(ratios.unsqueeze(0).to(device))

    layout_latents = torch.randn(
        (1, num_classes, layout_size, layout_size),
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )
    layout_latents = layout_latents * layout_scheduler.init_noise_sigma
    layout_scheduler.set_timesteps(args.num_inference_steps_layout, device=device)
    with torch.no_grad():
        for timestep in layout_scheduler.timesteps:
            noise_pred = layout_unet(layout_latents, timestep, class_labels=ratio_emb_layout).sample
            layout_latents = layout_scheduler.step(noise_pred, timestep, layout_latents).prev_sample

    layout_ids_64 = layout_latents.argmax(dim=1)
    layout_onehot_64 = F.one_hot(layout_ids_64, num_classes=num_classes).permute(0, 3, 1, 2).float()
    layout_onehot_512 = F.interpolate(layout_onehot_64, size=(args.image_size, args.image_size), mode="nearest")
    layout_ids_512 = F.interpolate(layout_ids_64.unsqueeze(1).float(), size=(args.image_size, args.image_size), mode="nearest")
    layout_ids_512 = layout_ids_512.squeeze(1).long()

    training_config = _load_json(controlnet_ckpt / "training_config.json") or {}
    base_model = args.base_model or training_config.get("base_model")
    if not base_model:
        raise ValueError("base_model is required. Pass --base_model or include it in controlnet training_config.json.")
    prompt = args.prompt or training_config.get("prompt") or "a high-resolution satellite image"

    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=weight_dtype)
    controlnet = ControlNetModel.from_pretrained(controlnet_ckpt / "controlnet", torch_dtype=weight_dtype)

    unet.register_to_config(class_embed_type="identity", num_class_embeds=None, projection_class_embeddings_input_dim=None)
    unet.class_embedding = torch.nn.Identity()
    if controlnet.class_embedding is None:
        controlnet.class_embedding = torch.nn.Identity()

    time_embed_dim = infer_time_embed_dim_from_config(unet.config.block_out_channels)
    ratio_projector = _load_ratio_projector(controlnet_ckpt, num_classes, time_embed_dim)
    film_gate = ResidualFiLMGate(time_embed_dim, n_down_blocks=len(controlnet.down_blocks))
    film_gate.load_state_dict(torch.load(controlnet_ckpt / "film_gate.bin", map_location="cpu"))

    if args.scheduler == "ddim":
        scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")
    else:
        scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)
    ratio_projector.to(device, dtype=weight_dtype)
    film_gate.to(device, dtype=weight_dtype)

    text_encoder.eval()
    vae.eval()
    unet.eval()
    controlnet.eval()
    ratio_projector.eval()
    film_gate.eval()

    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)

    ratio_emb = ratio_projector(ratios.unsqueeze(0).to(device))
    latents = torch.randn(
        (1, unet.config.in_channels, args.image_size // 8, args.image_size // 8),
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(args.num_inference_steps_image, device=device)

    with torch.no_grad():
        for timestep in scheduler.timesteps:
            down_samples, mid_sample = controlnet(
                latents,
                timestep,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=layout_onehot_512.to(device=device, dtype=weight_dtype),
                class_labels=ratio_emb,
                return_dict=False,
            )
            down_samples, mid_sample = film_gate(ratio_emb, down_samples, mid_sample)

            noise_pred = unet(
                latents,
                timestep,
                encoder_hidden_states=prompt_embeds,
                class_labels=ratio_emb,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
            ).sample
            latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    image = _vae_decode(vae, latents / vae.config.scaling_factor)[0]

    os.makedirs(args.save_dir, exist_ok=True)
    image_path = Path(args.save_dir) / "image.png"
    layout_path = Path(args.save_dir) / "layout.png"
    metadata_path = Path(args.save_dir) / "metadata.json"
    _save_uint8_rgb(image, image_path)
    _save_label_map(layout_ids_512[0], layout_path)

    metadata = {
        "prompt": prompt,
        "ratios": ratios.tolist(),
        "class_names": class_names,
        "layout_path": str(layout_path),
        "image_path": str(image_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Saved outputs under {args.save_dir}")


if __name__ == "__main__":
    main()
