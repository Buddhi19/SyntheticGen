#!/usr/bin/env python
# coding=utf-8

"""Sample (layout, image) pairs using layout DDPM + ControlNet with ratio conditioning."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)

try:
    from .dataset_loveda import build_palette
    from ..models.ratio_conditioning import (
        PerChannelResidualFiLMGate,
        RatioProjector,
        ResidualFiLMGate,
        infer_time_embed_dim_from_config,
    )
    from ..models.segmentation import SimpleSegNet
except ImportError:  # direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.scripts.dataset_loveda import build_palette
    from src.models.ratio_conditioning import (
        PerChannelResidualFiLMGate,
        RatioProjector,
        ResidualFiLMGate,
        infer_time_embed_dim_from_config,
    )
    from src.models.segmentation import SimpleSegNet


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample layout and image pairs.")
    parser.add_argument("--layout_ckpt", type=str, required=True, help="Layout DDPM checkpoint directory.")
    parser.add_argument("--controlnet_ckpt", type=str, required=True, help="ControlNet checkpoint directory.")
    parser.add_argument("--base_model", type=str, default=None, help="Base SD model path or ID.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional UNet LoRA dir (save_attn_procs).")
    parser.add_argument(
        "--lora_weight_name",
        type=str,
        default="pytorch_lora_weights.safetensors",
        help="LoRA weight filename inside lora_path.",
    )
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale (0 disables; 1 full).")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--ratios", type=str, default=None, help="Ratios as CSV or name:value pairs.")
    parser.add_argument("--ratios_json", type=str, default=None, help="JSON file with ratios list/dict.")
    parser.add_argument("--class_names_json", type=str, default=None, help="Optional class names JSON.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for image sampling.")
    parser.add_argument("--init_image", type=str, default=None, help="Optional init image for img2img editing.")
    parser.add_argument("--init_mask", type=str, default=None, help="Optional init mask (label map) for layout editing.")
    parser.add_argument("--strength_layout", type=float, default=0.8, help="Layout img2img strength (0..1).")
    parser.add_argument("--strength_image", type=float, default=0.8, help="Image img2img strength (0..1).")
    parser.add_argument("--ignore_index", type=int, default=255, help="Ignore index value for init masks.")
    parser.add_argument(
        "--mask_format",
        type=str,
        default="indexed",
        choices=["indexed", "loveda_raw"],
        help="Mask format: indexed (0..K-1 with ignore) or LoveDA raw (0=ignore, 1..K).",
    )
    parser.add_argument(
        "--seg_ckpt",
        type=str,
        default=None,
        help="Optional segmentation checkpoint for auto init mask (image-only editing).",
    )
    parser.add_argument(
        "--seg_arch",
        type=str,
        default="simple",
        choices=["simple"],
        help="Segmentation architecture for auto init mask.",
    )
    parser.add_argument("--hist_guidance_scale", type=float, default=0.0, help="Histogram guidance scale for layouts.")
    parser.add_argument("--hist_guidance_temp", type=float, default=1.0, help="Softmax temperature for histogram guidance.")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help="Optional CFG rescale to reduce overexposure/artifacts.",
    )
    parser.add_argument("--use_karras_sigmas", action="store_true")
    parser.add_argument(
        "--sampler",
        type=str,
        default="dpmpp_2m",
        choices=["ddim", "ddpm", "dpmpp_2m"],
        help="Image sampler for the second-stage diffusion model.",
    )
    parser.add_argument(
        "--scheduler",
        dest="sampler",
        type=str,
        default=argparse.SUPPRESS,
        choices=["ddim", "ddpm", "dpmpp_2m"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--num_inference_steps_layout", type=int, default=50)
    parser.add_argument("--num_inference_steps_image", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def rescale_noise_cfg(noise_cfg: torch.Tensor, noise_pred_text: torch.Tensor, guidance_rescale: float = 0.0) -> torch.Tensor:
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def _safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


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


def _colorize_labels(label_map: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    labels = label_map.detach().cpu().numpy().astype(np.int64)
    return palette[labels]


def _load_ratio_projector(checkpoint_dir: Path, num_classes: int, embed_dim: int) -> RatioProjector:
    projector = RatioProjector(num_classes, embed_dim)
    state_path = checkpoint_dir / "ratio_projector.bin"
    if not state_path.is_file():
        raise FileNotFoundError(f"ratio_projector not found at {state_path}")
    projector.load_state_dict(_safe_torch_load(state_path, map_location="cpu"))
    return projector


def _infer_num_down_residuals(controlnet) -> int:
    if hasattr(controlnet, "controlnet_down_blocks"):
        return len(controlnet.controlnet_down_blocks)
    if hasattr(controlnet, "config") and hasattr(controlnet.config, "down_block_types"):
        layers = getattr(controlnet.config, "layers_per_block", 1)
        return 1 + len(controlnet.config.down_block_types) * int(layers)
    return len(getattr(controlnet, "down_blocks", []))


def _ensure_identity_class_embedding(model) -> None:
    model.register_to_config(class_embed_type="identity", num_class_embeds=None, projection_class_embeddings_input_dim=None)
    model.class_embedding = torch.nn.Identity()


def _load_init_image(path: str, image_size: int) -> torch.Tensor:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor.unsqueeze(0)


def _load_init_mask(path: str, size: int) -> torch.Tensor:
    mask = Image.open(path)
    mask = ImageOps.exif_transpose(mask)
    if mask.mode not in {"L", "P"}:
        mask = mask.convert("L")
    mask = mask.resize((size, size), resample=Image.NEAREST)
    arr = np.asarray(mask, dtype=np.int64)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return torch.from_numpy(arr)


def _resize_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
    if mask.shape[-2:] == (size, size):
        return mask
    mask = mask.unsqueeze(0).float()
    mask = F.interpolate(mask, size=(size, size), mode="nearest").squeeze(0)
    return mask.long()


def _onehot_from_mask(
    mask: torch.Tensor, num_classes: int, ignore_index: Optional[int], mask_format: str
) -> torch.Tensor:
    if mask_format == "loveda_raw":
        mask = mask.clone()
        ignore = mask == 0
        mask = mask - 1
        if ignore_index is None:
            ignore_index = -1
        mask[ignore] = ignore_index
    elif mask_format != "indexed":
        raise ValueError(f"Unsupported mask_format: {mask_format}")

    if ignore_index is None:
        valid = torch.ones_like(mask, dtype=torch.bool)
    else:
        valid = mask != ignore_index
    safe = mask.clone()
    safe[~valid] = 0
    safe = safe.clamp(min=0, max=num_classes - 1)
    onehot = F.one_hot(safe.long(), num_classes=num_classes).permute(2, 0, 1).float()
    return onehot * valid.unsqueeze(0).float()


def _load_segmentation_model(seg_arch: str, num_classes: int, ckpt_path: Optional[str], device: torch.device):
    if seg_arch != "simple":
        raise ValueError(f"Unsupported seg_arch: {seg_arch}")
    if not ckpt_path:
        raise ValueError("--seg_ckpt is required for image-only editing.")
    model = SimpleSegNet(num_classes)
    state = _safe_torch_load(Path(ckpt_path), map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
    model.load_state_dict(state)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def _predict_mask_from_image(model, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        logits = model(image.to(device=device, dtype=torch.float32))
        preds = torch.argmax(logits, dim=1)
    return preds[0].detach().cpu()


def _get_strength_timesteps(num_steps: int, strength: float) -> Tuple[int, int]:
    strength = min(max(strength, 0.0), 1.0)
    init_timestep = int(num_steps * strength)
    init_timestep = min(init_timestep, num_steps)
    t_start = max(num_steps - init_timestep, 0)
    return init_timestep, t_start


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

    init_image_tensor = None
    if args.init_image:
        init_image_tensor = _load_init_image(args.init_image, args.image_size)

    init_mask_tensor = None
    mask_format = args.mask_format
    if args.init_mask:
        init_mask_tensor = _load_init_mask(args.init_mask, layout_size)
    elif init_image_tensor is not None:
        seg_model = _load_segmentation_model(args.seg_arch, num_classes, args.seg_ckpt, device)
        seg_mask = _predict_mask_from_image(seg_model, init_image_tensor, device)
        init_mask_tensor = _resize_mask(seg_mask, layout_size)
        mask_format = "indexed"

    ratios_device = ratios.to(device=device, dtype=weight_dtype)
    ratio_emb_layout = layout_ratio_projector(ratios_device.unsqueeze(0)).to(dtype=weight_dtype)
    layout_scheduler.set_timesteps(args.num_inference_steps_layout, device=device)
    if init_mask_tensor is not None:
        init_mask = init_mask_tensor.to(device)
        layout_onehot_init = _onehot_from_mask(init_mask, num_classes, args.ignore_index, mask_format)
        layout_onehot_init = layout_onehot_init.to(device, dtype=weight_dtype)
        layout_latents = layout_onehot_init.unsqueeze(0) * 2.0 - 1.0
        _, t_start = _get_strength_timesteps(args.num_inference_steps_layout, args.strength_layout)
        timesteps = layout_scheduler.timesteps[t_start:]
        if len(timesteps) == 0:
            timesteps = []
        else:
            noise = torch.randn(
                layout_latents.shape,
                generator=generator,
                device=device,
                dtype=layout_latents.dtype,
            )
            layout_latents = layout_scheduler.add_noise(layout_latents, noise, timesteps[0])
    else:
        layout_latents = torch.randn(
            (1, num_classes, layout_size, layout_size),
            generator=generator,
            device=device,
            dtype=weight_dtype,
        )
        layout_latents = layout_latents * layout_scheduler.init_noise_sigma
        timesteps = layout_scheduler.timesteps

    alphas_cumprod = layout_scheduler.alphas_cumprod.to(device=device, dtype=layout_latents.dtype)
    with torch.no_grad():
        for timestep in timesteps:
            noise_pred = layout_unet(layout_latents, timestep, class_labels=ratio_emb_layout).sample
            if args.hist_guidance_scale > 0:
                t_index = timestep.long() if isinstance(timestep, torch.Tensor) else int(timestep)
                alpha_prod = alphas_cumprod[t_index].view(1, 1, 1, 1)
                sqrt_alpha = alpha_prod.sqrt()
                sqrt_one_minus = (1 - alpha_prod).sqrt()
                x0_pred = (layout_latents - sqrt_one_minus * noise_pred) / sqrt_alpha
                probs = torch.softmax(x0_pred / args.hist_guidance_temp, dim=1)
                r_hat = probs.mean(dim=(2, 3))
                delta = (ratios_device - r_hat).view(1, num_classes, 1, 1)
            layout_latents = layout_scheduler.step(noise_pred, timestep, layout_latents).prev_sample
            if args.hist_guidance_scale > 0:
                layout_latents = layout_latents + args.hist_guidance_scale * delta

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
    palette = build_palette(class_names, num_classes, dataset=training_config.get("dataset"))

    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=weight_dtype)
    if args.lora_path is not None:
        unet.load_attn_procs(args.lora_path, weight_name=args.lora_weight_name)
        logger.info("Loaded UNet LoRA from %s (%s)", args.lora_path, args.lora_weight_name)
    controlnet = ControlNetModel.from_pretrained(controlnet_ckpt / "controlnet", torch_dtype=weight_dtype)

    _ensure_identity_class_embedding(unet)
    _ensure_identity_class_embedding(controlnet)

    time_embed_dim = infer_time_embed_dim_from_config(unet.config.block_out_channels)
    ratio_projector = _load_ratio_projector(controlnet_ckpt, num_classes, time_embed_dim)
    gate_state = _safe_torch_load(controlnet_ckpt / "film_gate.bin", map_location="cpu")
    if isinstance(gate_state, dict) and "proj.weight" in gate_state:
        film_gate = ResidualFiLMGate(time_embed_dim, n_down_blocks=_infer_num_down_residuals(controlnet))
    elif isinstance(gate_state, dict) and any(str(k).startswith("down_mlps.") for k in gate_state.keys()):
        down_channels = [int(block.out_channels) for block in controlnet.controlnet_down_blocks]
        mid_channels = int(controlnet.controlnet_mid_block.out_channels)
        film_gate = PerChannelResidualFiLMGate(
            time_embed_dim,
            down_channels=down_channels,
            mid_channels=mid_channels,
            init_zero=False,
        )
    else:
        raise ValueError(f"Unrecognized FiLM gate checkpoint format: {controlnet_ckpt / 'film_gate.bin'}")
    film_gate.load_state_dict(gate_state)

    if args.sampler == "dpmpp_2m":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            base_model,
            subfolder="scheduler",
            use_karras_sigmas=args.use_karras_sigmas,
        )
    elif args.sampler == "ddpm":
        scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    else:
        scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")

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

    uncond_inputs = tokenizer(
        [""],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]
    uncond_embeds = uncond_embeds.to(dtype=weight_dtype)

    ratio_emb = ratio_projector(ratios_device.unsqueeze(0)).to(dtype=weight_dtype)
    layout_cond = layout_onehot_512.to(device=device, dtype=weight_dtype)
    layout_uncond = torch.zeros_like(layout_cond)
    ratio_uncond = torch.zeros_like(ratio_emb)
    scheduler.set_timesteps(args.num_inference_steps_image, device=device)
    lora_scale = args.lora_scale if args.lora_scale is not None else 1.0
    lora_cross_kwargs = None
    if args.lora_path is not None and float(lora_scale) != 1.0:
        lora_cross_kwargs = {"scale": float(lora_scale)}
    if args.init_image:
        if init_image_tensor is None:
            init_image_tensor = _load_init_image(args.init_image, args.image_size)
        init_image = init_image_tensor.to(device=device, dtype=weight_dtype)
        with torch.no_grad():
            latents = vae.encode(init_image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        _, t_start = _get_strength_timesteps(args.num_inference_steps_image, args.strength_image)
        timesteps = scheduler.timesteps[t_start:]
        if len(timesteps) == 0:
            timesteps = []
        else:
            noise = torch.randn(
                latents.shape,
                generator=generator,
                device=device,
                dtype=latents.dtype,
            )
            latents = scheduler.add_noise(latents, noise, timesteps[0])
    else:
        latents = torch.randn(
            (1, unet.config.in_channels, args.image_size // 8, args.image_size // 8),
            generator=generator,
            device=device,
            dtype=weight_dtype,
        )
        latents = latents * scheduler.init_noise_sigma
        timesteps = scheduler.timesteps

    do_cfg = args.guidance_scale is not None and float(args.guidance_scale) != 1.0
    with torch.no_grad():
        for timestep in timesteps:
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2, dim=0)
                encoder_hidden_states = torch.cat([uncond_embeds, prompt_embeds], dim=0)
                controlnet_cond = torch.cat([layout_uncond, layout_cond], dim=0)
                class_labels = torch.cat([ratio_uncond, ratio_emb], dim=0)
            else:
                latent_model_input = latents
                encoder_hidden_states = prompt_embeds
                controlnet_cond = layout_cond
                class_labels = ratio_emb

            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

            down_samples, mid_sample = controlnet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                class_labels=class_labels,
                return_dict=False,
            )
            down_samples, mid_sample = film_gate(class_labels, down_samples, mid_sample)

            noise_pred = unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=class_labels,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
                cross_attention_kwargs=lora_cross_kwargs,
            ).sample

            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_cfg = noise_uncond + float(args.guidance_scale) * (noise_text - noise_uncond)
                if args.guidance_rescale and args.guidance_rescale > 0:
                    noise_cfg = rescale_noise_cfg(noise_cfg, noise_text, guidance_rescale=float(args.guidance_rescale))
                noise_pred = noise_cfg

            latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    image = _vae_decode(vae, latents / vae.config.scaling_factor)[0]

    os.makedirs(args.save_dir, exist_ok=True)
    image_path = Path(args.save_dir) / "image.png"
    layout_path = Path(args.save_dir) / "layout.png"
    layout_color_path = Path(args.save_dir) / "layout_color.png"
    metadata_path = Path(args.save_dir) / "metadata.json"
    _save_uint8_rgb(image, image_path)
    _save_label_map(layout_ids_512[0], layout_path)
    layout_color = _colorize_labels(layout_ids_512[0], palette)
    Image.fromarray(layout_color, mode="RGB").save(layout_color_path)

    metadata = {
        "prompt": prompt,
        "ratios": ratios.tolist(),
        "class_names": class_names,
        "layout_path": str(layout_path),
        "layout_color_path": str(layout_color_path),
        "image_path": str(image_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Saved outputs under {args.save_dir}")


if __name__ == "__main__":
    main()
