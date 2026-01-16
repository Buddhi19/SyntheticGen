#!/usr/bin/env python
# coding=utf-8

"""Sample (layout, image) pairs using layout DDPM + ControlNet with ratio conditioning."""

import argparse
import json
import logging
import os
import re
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
    from .dataset_loveda import DEFAULT_LOVEDA_CLASS_NAMES, build_palette
    from .config_utils import apply_config
    from ..models.d3pm import D3PMScheduler
    from ..models.ratio_conditioning import (
        PerChannelResidualFiLMGate,
        RatioProjector,
        ResidualFiLMGate,
        build_ratio_projector_from_state_dict,
        infer_time_embed_dim_from_config,
    )
    from ..models.segmentation import SimpleSegNet
except ImportError:  # direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.scripts.dataset_loveda import DEFAULT_LOVEDA_CLASS_NAMES, build_palette
    from src.scripts.config_utils import apply_config
    from src.models.d3pm import D3PMScheduler
    from src.models.ratio_conditioning import (
        PerChannelResidualFiLMGate,
        RatioProjector,
        ResidualFiLMGate,
        build_ratio_projector_from_state_dict,
        infer_time_embed_dim_from_config,
    )
    from src.models.segmentation import SimpleSegNet


logger = logging.getLogger(__name__)


def parse_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML/JSON config file; values act as argparse defaults.",
    )
    cfg_args, remaining = base_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Sample layout and image pairs.", parents=[base_parser])
    parser.add_argument("--layout_ckpt", type=str, default=None, help="Layout DDPM checkpoint directory.")
    parser.add_argument(
        "--layout_checkpoint",
        type=int,
        default=None,
        help="Optional layout checkpoint step when --layout_ckpt points to a run dir with checkpoint-XXXXX subfolders.",
    )
    parser.add_argument(
        "--layout_size",
        type=int,
        default=None,
        help="Layout diffusion size (square). Used when loading an intermediate checkpoint (checkpoint-XXXXX).",
    )
    parser.add_argument(
        "--layout_diffusion_type",
        type=str,
        default=None,
        choices=["d3pm", "ddpm"],
        help="Layout diffusion type when it cannot be inferred from the checkpoint (e.g., intermediate checkpoints).",
    )
    parser.add_argument(
        "--layout_num_train_timesteps",
        type=int,
        default=1000,
        help="Layout diffusion training timesteps (used for intermediate checkpoints).",
    )
    parser.add_argument(
        "--layout_beta_start",
        type=float,
        default=1e-4,
        help="D3PM beta_start (used for intermediate checkpoints when --layout_diffusion_type=d3pm).",
    )
    parser.add_argument(
        "--layout_beta_end",
        type=float,
        default=0.02,
        help="D3PM beta_end (used for intermediate checkpoints when --layout_diffusion_type=d3pm).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="urban",
        choices=["urban", "rural"],
        help="Domain conditioning for Stage-A layout generation.",
    )
    parser.add_argument(
        "--domain_cond_scale",
        type=float,
        default=1.0,
        help="Scale for domain embedding during Stage-A layout generation.",
    )
    parser.add_argument(
        "--controlnet_ckpt",
        type=str,
        default=None,
        help="ControlNet checkpoint directory (export dir or training checkpoint-XXXXX).",
    )
    parser.add_argument("--base_model", type=str, default=None, help="Base SD model path or ID.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional UNet LoRA dir (save_attn_procs).")
    parser.add_argument(
        "--lora_weight_name",
        type=str,
        default="pytorch_lora_weights.safetensors",
        help="LoRA weight filename inside lora_path.",
    )
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale (0 disables; 1 full).")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save outputs.")
    parser.add_argument(
        "--ratios",
        type=str,
        default=None,
        help="Ratios as CSV or name:value pairs (supports partial specs).",
    )
    parser.add_argument(
        "--ratios_json",
        type=str,
        default=None,
        help="JSON file with ratios list/dict (supports partial specs).",
    )
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
    parser.add_argument(
        "--control_scale",
        type=float,
        default=2.0,
        help="Scale factor for ControlNet residuals (like controlnet_conditioning_scale).",
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
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    if cfg_args.config:
        apply_config(parser, cfg_args.config)
    args = parser.parse_args(remaining)
    args.config = cfg_args.config

    missing = [name for name in ["layout_ckpt", "controlnet_ckpt", "save_dir"] if getattr(args, name) in (None, "")]
    if missing:
        parser.error(f"Missing required arguments: {', '.join('--' + x for x in missing)} (pass them directly or via --config).")

    return args


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


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load .safetensors checkpoints; install it with `pip install safetensors`."
            ) from exc
        return safe_load_file(str(path))
    return _safe_torch_load(path, map_location="cpu")


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
    if int(num_classes) == int(len(DEFAULT_LOVEDA_CLASS_NAMES)):
        # When sampling from intermediate checkpoints, training_config/class_names.json may not exist yet.
        # Defaulting to LoveDA class names avoids ratio-name KeyErrors like "building:0.4".
        return [str(x) for x in DEFAULT_LOVEDA_CLASS_NAMES]
    return [f"class_{i}" for i in range(num_classes)]


def _parse_ratio_constraints(
    ratios_str: Optional[str],
    ratios_json: Optional[str],
    num_classes: int,
    class_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (ratio_values, known_mask), both float32 tensors of shape (K,)."""
    name_to_idx = {str(name).strip().lower(): i for i, name in enumerate(class_names)}

    values = [0.0] * num_classes
    known = [0.0] * num_classes

    if ratios_json:
        data = _load_json(Path(ratios_json))
        if isinstance(data, list):
            if len(data) > num_classes:
                raise ValueError(f"Expected at most {num_classes} ratios, got {len(data)}.")
            for i, x in enumerate(data):
                values[i] = float(x)
                known[i] = 1.0
        elif isinstance(data, dict):
            for key, value in data.items():
                key_str = str(key).strip()
                idx = int(key_str) if key_str.isdigit() else name_to_idx[key_str.lower()]
                values[idx] = float(value)
                known[idx] = 1.0
        else:
            raise ValueError("ratios_json must be a list or dict.")
    elif ratios_str:
        ratios_str = str(ratios_str)
        if ":" in ratios_str:
            for chunk in ratios_str.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                name, value = chunk.split(":", maxsplit=1)
                name = name.strip()
                idx = int(name) if name.isdigit() else name_to_idx[name.lower()]
                values[idx] = float(value)
                known[idx] = 1.0
        else:
            raw = [x for x in ratios_str.split(",") if x.strip() != ""]
            if len(raw) > num_classes:
                raise ValueError(f"Expected at most {num_classes} ratios, got {len(raw)}.")
            for i, x in enumerate(raw):
                values[i] = float(x)
                known[i] = 1.0
    else:
        raise ValueError("Provide --ratios or --ratios_json.")

    values_t = torch.tensor(values, dtype=torch.float32)
    known_t = torch.tensor(known, dtype=torch.float32)
    if torch.any(values_t < 0):
        raise ValueError("Ratios must be non-negative.")
    if known_t.sum().item() <= 0:
        raise ValueError("At least one ratio must be specified.")
    known_sum = float((values_t * known_t).sum().item())
    if known_t.sum().item() < num_classes and known_sum > 1.0 + 1e-6:
        raise ValueError("Specified ratios sum to more than 1.0.")
    return values_t, known_t


def _impute_full_ratios(ratio_values: torch.Tensor, known_mask: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    """Returns a full ratio vector of shape (K,) that sums to 1 (fills unknowns randomly)."""
    ratio_values = ratio_values.detach().cpu().to(dtype=torch.float32)
    known_mask = known_mask.detach().cpu().to(dtype=torch.float32)
    if ratio_values.ndim != 1 or known_mask.ndim != 1:
        raise ValueError("ratio_values/known_mask must be 1D tensors.")
    if ratio_values.shape[0] != known_mask.shape[0]:
        raise ValueError("ratio_values and known_mask must have the same length.")

    if torch.all(known_mask > 0):
        total = ratio_values.sum()
        if total <= 0:
            raise ValueError("Ratios must sum to a positive value.")
        return ratio_values / total

    known_sum = (ratio_values * known_mask).sum()
    if known_sum > 1.0 + 1e-6:
        raise ValueError("Specified ratios sum to more than 1.0.")
    remaining = float(max(0.0, 1.0 - float(known_sum.item())))

    full = ratio_values * known_mask
    missing = torch.nonzero(known_mask <= 0, as_tuple=False).flatten()
    if missing.numel() > 0 and remaining > 0:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        rand = torch.rand(int(missing.numel()), generator=generator)
        rand = rand / rand.sum().clamp(min=1e-8)
        full[missing] = rand * remaining

    total = full.sum()
    if total <= 0:
        raise ValueError("Ratios must sum to a positive value.")
    return full / total


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
    state_path = checkpoint_dir / "ratio_projector.bin"
    if not state_path.is_file():
        raise FileNotFoundError(f"ratio_projector not found at {state_path}")
    state = _load_state_dict(state_path)
    projector = build_ratio_projector_from_state_dict(state, num_classes=num_classes, embed_dim=embed_dim)
    projector.load_state_dict(state)
    return projector


def _list_checkpoint_steps(run_dir: Path) -> List[int]:
    if not run_dir.is_dir():
        return []
    steps: List[int] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.match(r"^checkpoint-(\d+)$", child.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def _resolve_run_and_checkpoint_dir(base_dir: Path, checkpoint: Optional[int], kind: str) -> Tuple[Path, Path]:
    if base_dir.name.startswith("checkpoint-"):
        if checkpoint is not None:
            logger.warning("--%s_checkpoint ignored because --%s_ckpt is already a checkpoint dir: %s", kind, kind, base_dir)
        return base_dir.parent, base_dir

    steps = _list_checkpoint_steps(base_dir)
    if steps:
        run_dir = base_dir
        step = int(checkpoint) if checkpoint is not None else steps[-1]
        ckpt_dir = run_dir / f"checkpoint-{step}"
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"{kind} checkpoint not found: {ckpt_dir}")
        if checkpoint is None:
            logger.info("Resolved %s checkpoint to latest: %s", kind, ckpt_dir)
        else:
            logger.info("Resolved %s checkpoint to: %s", kind, ckpt_dir)
        return run_dir, ckpt_dir

    return base_dir, base_dir


def _strip_state_dict_prefix(state: Dict[str, torch.Tensor], required_key: str) -> Dict[str, torch.Tensor]:
    if required_key in state:
        return state
    candidates = [k for k in state.keys() if k.endswith(required_key)]
    if not candidates:
        return state
    chosen = min(candidates, key=len)
    prefix = chosen[: -len(required_key)]
    stripped = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if stripped:
        return stripped
    return state


def _infer_unet2d_config_from_state_dict(state: Dict[str, torch.Tensor], sample_size: int) -> Dict:
    state = _strip_state_dict_prefix(state, "conv_in.weight")
    if "conv_in.weight" not in state:
        raise KeyError("Could not find conv_in.weight in UNet checkpoint state dict.")

    conv_in = state["conv_in.weight"]
    in_channels = int(conv_in.shape[1])

    conv_out = state.get("conv_out.weight")
    out_channels = int(conv_out.shape[0]) if conv_out is not None else int(in_channels)

    down_block_indices = set()
    resnet_indices = set()
    for key in state.keys():
        m = re.match(r"^down_blocks\.(\d+)\.", key)
        if m:
            down_block_indices.add(int(m.group(1)))
        m = re.match(r"^down_blocks\.0\.resnets\.(\d+)\.", key)
        if m:
            resnet_indices.add(int(m.group(1)))
    if not down_block_indices:
        raise ValueError("Could not infer down_blocks.* from UNet checkpoint state dict.")

    num_down_blocks = max(down_block_indices) + 1
    layers_per_block = (max(resnet_indices) + 1) if resnet_indices else 1

    block_out_channels: List[int] = []
    for i in range(num_down_blocks):
        for candidate in (
            f"down_blocks.{i}.resnets.0.conv2.weight",
            f"down_blocks.{i}.resnets.0.conv1.weight",
            f"down_blocks.{i}.resnets.0.norm2.weight",
            f"down_blocks.{i}.resnets.0.norm1.weight",
        ):
            tensor = state.get(candidate)
            if tensor is not None:
                block_out_channels.append(int(tensor.shape[0]))
                break
        else:
            raise ValueError(f"Could not infer block_out_channels for down_blocks.{i} from checkpoint state dict.")

    return {
        "sample_size": int(sample_size),
        "in_channels": int(in_channels),
        "out_channels": int(out_channels),
        "layers_per_block": int(layers_per_block),
        "block_out_channels": tuple(int(x) for x in block_out_channels),
        "down_block_types": tuple("DownBlock2D" for _ in range(num_down_blocks)),
        "up_block_types": tuple("UpBlock2D" for _ in range(num_down_blocks)),
        "class_embed_type": "identity",
    }


def _load_layout_from_checkpoint(
    ckpt_dir: Path,
    device: torch.device,
    layout_size: int,
    diffusion_type: str,
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
) -> Tuple[UNet2DModel, RatioProjector, object, bool]:
    unet_state_path = None
    for candidate in ("model.safetensors", "pytorch_model.bin", "model.bin"):
        path = ckpt_dir / candidate
        if path.is_file():
            unet_state_path = path
            break
    if unet_state_path is None:
        raise FileNotFoundError(f"Could not find UNet weights under {ckpt_dir} (expected model.safetensors).")

    unet_state = _load_state_dict(unet_state_path)
    unet_state = _strip_state_dict_prefix(unet_state, "conv_in.weight")
    unet_config = _infer_unet2d_config_from_state_dict(unet_state, sample_size=layout_size)
    layout_unet = UNet2DModel(**unet_config)
    layout_unet.load_state_dict(unet_state)

    num_classes = int(layout_unet.config.in_channels)
    time_embed_dim_layout = infer_time_embed_dim_from_config(layout_unet.config.block_out_channels)

    ratio_state_path = None
    for candidate in ("model_1.safetensors", "pytorch_model_1.bin", "model_1.bin"):
        path = ckpt_dir / candidate
        if path.is_file():
            ratio_state_path = path
            break
    if ratio_state_path is None:
        raise FileNotFoundError(f"Could not find ratio projector weights under {ckpt_dir} (expected model_1.safetensors).")
    ratio_state = _load_state_dict(ratio_state_path)
    ratio_state = _strip_state_dict_prefix(ratio_state, "net.0.weight")
    ratio_projector = build_ratio_projector_from_state_dict(ratio_state, num_classes=num_classes, embed_dim=time_embed_dim_layout)
    ratio_projector.load_state_dict(ratio_state)

    d3pm_cfg = ckpt_dir / "d3pm_config.json"
    if d3pm_cfg.is_file():
        layout_scheduler = D3PMScheduler.from_config(d3pm_cfg, device=device)
        layout_is_d3pm = True
    else:
        if diffusion_type == "d3pm":
            layout_scheduler = D3PMScheduler(
                num_classes=num_classes,
                num_timesteps=int(num_train_timesteps),
                beta_start=float(beta_start),
                beta_end=float(beta_end),
                device=device,
            )
            layout_is_d3pm = True
        elif diffusion_type == "ddpm":
            layout_scheduler = DDPMScheduler(num_train_timesteps=int(num_train_timesteps))
            layout_is_d3pm = False
        else:
            raise ValueError(
                "Could not infer layout diffusion type from checkpoint; pass --layout_diffusion_type (d3pm|ddpm)."
            )

    return layout_unet, ratio_projector, layout_scheduler, layout_is_d3pm


def _load_domain_embedding(layout_run_dir: Path, layout_ckpt_dir: Path) -> Optional[torch.nn.Embedding]:
    candidates: List[Path] = []
    run_path = layout_run_dir / "domain_embed.bin"
    if run_path.is_file():
        candidates.append(run_path)
    for candidate in ("model_2.safetensors", "pytorch_model_2.bin", "model_2.bin"):
        ckpt_path = layout_ckpt_dir / candidate
        if ckpt_path.is_file():
            candidates.append(ckpt_path)
            break

    for path in candidates:
        state = _load_state_dict(path)
        state = _strip_state_dict_prefix(state, "weight")
        weight = state.get("weight")
        if weight is None or getattr(weight, "ndim", 0) != 2:
            raise ValueError(f"Invalid domain embedding checkpoint: {path}")
        emb = torch.nn.Embedding(int(weight.shape[0]), int(weight.shape[1]))
        emb.load_state_dict(state)
        return emb
    return None


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
    batch_size = int(args.batch_size)
    if batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    layout_ckpt = Path(args.layout_ckpt)
    controlnet_ckpt = Path(args.controlnet_ckpt)
    if not layout_ckpt.exists():
        raise FileNotFoundError(f"Layout checkpoint not found: {layout_ckpt}")
    if not controlnet_ckpt.exists():
        raise FileNotFoundError(f"ControlNet checkpoint not found: {controlnet_ckpt}")
    controlnet_root = controlnet_ckpt
    controlnet_state_ckpt = None
    if (controlnet_ckpt / "model.safetensors").is_file():
        controlnet_root = controlnet_ckpt.parent
        controlnet_state_ckpt = controlnet_ckpt

    layout_run_dir, layout_load_dir = _resolve_run_and_checkpoint_dir(layout_ckpt, args.layout_checkpoint, kind="layout")

    if (layout_load_dir / "layout_unet").is_dir():
        layout_unet = UNet2DModel.from_pretrained(layout_load_dir / "layout_unet")
        num_classes = layout_unet.config.in_channels
        layout_size = layout_unet.config.sample_size
        time_embed_dim_layout = infer_time_embed_dim_from_config(layout_unet.config.block_out_channels)
        layout_ratio_projector = _load_ratio_projector(layout_load_dir, num_classes, time_embed_dim_layout)
        d3pm_config_path = layout_load_dir / "d3pm_config.json"
        if d3pm_config_path.is_file():
            layout_scheduler = D3PMScheduler.from_config(d3pm_config_path, device=device)
            layout_is_d3pm = True
        else:
            layout_scheduler = DDPMScheduler.from_pretrained(layout_load_dir / "scheduler")
            layout_is_d3pm = False
    else:
        inferred_layout_size = args.layout_size
        if inferred_layout_size is None:
            if args.image_size is not None and int(args.image_size) % 4 == 0:
                inferred_layout_size = int(args.image_size) // 4
                logger.info("Inferred --layout_size=%d from --image_size=%d (override with --layout_size).", inferred_layout_size, args.image_size)
            else:
                inferred_layout_size = 256
                logger.info("Defaulting --layout_size=%d for checkpoint loading (override with --layout_size).", inferred_layout_size)
        layout_unet, layout_ratio_projector, layout_scheduler, layout_is_d3pm = _load_layout_from_checkpoint(
            layout_load_dir,
            device=device,
            layout_size=int(inferred_layout_size),
            diffusion_type=str(args.layout_diffusion_type) if args.layout_diffusion_type is not None else "",
            num_train_timesteps=int(args.layout_num_train_timesteps),
            beta_start=float(args.layout_beta_start),
            beta_end=float(args.layout_beta_end),
        )
        num_classes = layout_unet.config.in_channels
        layout_size = int(inferred_layout_size)

    class_names = _load_class_names(args, layout_run_dir, controlnet_root, num_classes)
    domain_embed = _load_domain_embedding(layout_run_dir, layout_load_dir)
    ratios_requested, ratios_known_mask = _parse_ratio_constraints(args.ratios, args.ratios_json, num_classes, class_names)
    ratios_full = _impute_full_ratios(ratios_requested, ratios_known_mask, seed=args.seed)

    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    layout_unet.to(device, dtype=weight_dtype)
    layout_ratio_projector.to(device, dtype=weight_dtype)
    layout_unet.eval()
    layout_ratio_projector.eval()
    if domain_embed is not None:
        domain_embed.to(device, dtype=weight_dtype)
        domain_embed.eval()

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

    layout_ratio_input_dim = getattr(layout_ratio_projector, "input_dim", num_classes)
    if layout_ratio_input_dim == num_classes:
        ratios_layout = ratios_full
        ratios_layout_mask = torch.ones_like(ratios_full)
    else:
        if torch.all(ratios_known_mask > 0):
            ratios_layout = ratios_full
            ratios_layout_mask = torch.ones_like(ratios_full)
        else:
            ratios_layout = ratios_requested
            ratios_layout_mask = ratios_known_mask
    ratios_layout_batch = ratios_layout.unsqueeze(0).repeat(batch_size, 1)
    ratios_layout_mask_batch = ratios_layout_mask.unsqueeze(0).repeat(batch_size, 1)
    ratios_layout_device = ratios_layout_batch.to(device=device, dtype=weight_dtype)
    ratios_layout_mask_device = ratios_layout_mask_batch.to(device=device, dtype=weight_dtype)
    if layout_ratio_input_dim == num_classes:
        ratio_emb_layout = layout_ratio_projector(ratios_layout_device).to(dtype=weight_dtype)
    else:
        ratio_emb_layout = layout_ratio_projector(ratios_layout_device, ratios_layout_mask_device).to(dtype=weight_dtype)
    cond_emb_layout = ratio_emb_layout
    if domain_embed is not None:
        dom_map = {"urban": 0, "rural": 1}
        dom_id = int(dom_map[str(args.domain).lower()])
        if dom_id >= int(domain_embed.num_embeddings):
            raise ValueError(
                f"Domain '{args.domain}' not supported by domain_embed (num_domains={domain_embed.num_embeddings})."
            )
        dom = torch.full((batch_size,), dom_id, device=device, dtype=torch.long)
        dom_emb = domain_embed(dom).to(dtype=cond_emb_layout.dtype)
        cond_emb_layout = cond_emb_layout + float(args.domain_cond_scale) * dom_emb
    layout_scheduler.set_timesteps(args.num_inference_steps_layout, device=device)
    if init_mask_tensor is not None:
        init_mask = init_mask_tensor.to(device)
        layout_onehot_init = _onehot_from_mask(init_mask, num_classes, args.ignore_index, mask_format)
        layout_onehot_init = layout_onehot_init.to(device, dtype=weight_dtype)
        layout_ids_init = layout_onehot_init.argmax(dim=0).to(dtype=torch.long)
        layout_ids_init = layout_ids_init.unsqueeze(0).repeat(batch_size, 1, 1)
        layout_onehot_init = layout_onehot_init.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        if layout_is_d3pm:
            layout_latents = layout_ids_init
        else:
            layout_latents = layout_onehot_init * 2.0 - 1.0
        _, t_start = _get_strength_timesteps(args.num_inference_steps_layout, args.strength_layout)
        timesteps = layout_scheduler.timesteps[t_start:]
        if len(timesteps) == 0:
            timesteps = []
        elif layout_is_d3pm:
            t_init = int(timesteps[0].item()) if isinstance(timesteps[0], torch.Tensor) else int(timesteps[0])
            t_init_batch = torch.full((batch_size,), t_init, device=device, dtype=torch.long)
            layout_latents = layout_scheduler.q_sample(
                layout_latents,
                t_init_batch,
                generator=generator,
            )
        else:
            noise = torch.randn(
                layout_latents.shape,
                generator=generator,
                device=device,
                dtype=layout_latents.dtype,
            )
            layout_latents = layout_scheduler.add_noise(layout_latents, noise, timesteps[0])
    else:
        timesteps = layout_scheduler.timesteps
        if layout_is_d3pm:
            layout_latents = torch.randint(
                0,
                int(num_classes),
                size=(batch_size, int(layout_size), int(layout_size)),
                generator=generator,
                device=device,
                dtype=torch.long,
            )
        else:
            layout_latents = torch.randn(
                (batch_size, num_classes, layout_size, layout_size),
                generator=generator,
                device=device,
                dtype=weight_dtype,
            )
            layout_latents = layout_latents * layout_scheduler.init_noise_sigma

    if layout_is_d3pm:
        with torch.no_grad():
            for idx, t in enumerate(timesteps):
                t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
                t_prev = int(timesteps[idx + 1].item()) if idx + 1 < len(timesteps) else 0
                t_batch = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
                t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                x_t_onehot = F.one_hot(layout_latents, num_classes=num_classes).permute(0, 3, 1, 2).float()
                x_t_onehot = x_t_onehot.to(device=device, dtype=weight_dtype)
                logits_x0 = layout_unet(
                    x_t_onehot, t_batch, class_labels=cond_emb_layout
                ).sample
                if args.hist_guidance_scale > 0:
                    probs = torch.softmax(logits_x0 / float(args.hist_guidance_temp), dim=1)
                    r_hat = probs.mean(dim=(2, 3))
                    delta = ((ratios_layout_device - r_hat) * ratios_layout_mask_device).view(-1, num_classes, 1, 1)
                    logits_x0 = logits_x0 + float(args.hist_guidance_scale) * delta.to(dtype=logits_x0.dtype)
                logp_prev = layout_scheduler.p_theta_posterior_logprobs(
                    layout_latents,
                    t=t_batch,
                    logits_x0=logits_x0,
                    t_prev=t_prev_batch,
                )
                layout_latents = layout_scheduler.sample_from_logprobs(logp_prev, generator=generator)
        layout_ids_64 = layout_latents
    else:
        alphas_cumprod = layout_scheduler.alphas_cumprod.to(device=device, dtype=layout_latents.dtype)
        with torch.no_grad():
            for timestep in timesteps:
                noise_pred = layout_unet(layout_latents, timestep, class_labels=cond_emb_layout).sample
                if args.hist_guidance_scale > 0:
                    t_index = timestep.long() if isinstance(timestep, torch.Tensor) else int(timestep)
                    alpha_prod = alphas_cumprod[t_index].view(1, 1, 1, 1)
                    sqrt_alpha = alpha_prod.sqrt()
                    sqrt_one_minus = (1 - alpha_prod).sqrt()
                    x0_pred = (layout_latents - sqrt_one_minus * noise_pred) / sqrt_alpha
                    probs = torch.softmax(x0_pred / args.hist_guidance_temp, dim=1)
                    r_hat = probs.mean(dim=(2, 3))
                    delta = ((ratios_layout_device - r_hat) * ratios_layout_mask_device).view(-1, num_classes, 1, 1)
                layout_latents = layout_scheduler.step(noise_pred, timestep, layout_latents).prev_sample
                if args.hist_guidance_scale > 0:
                    layout_latents = layout_latents + args.hist_guidance_scale * delta
        layout_ids_64 = layout_latents.argmax(dim=1)
    layout_onehot_64 = F.one_hot(layout_ids_64, num_classes=num_classes).permute(0, 3, 1, 2).float()
    layout_onehot_512 = F.interpolate(layout_onehot_64, size=(args.image_size, args.image_size), mode="nearest")
    layout_ids_512 = F.interpolate(layout_ids_64.unsqueeze(1).float(), size=(args.image_size, args.image_size), mode="nearest")
    layout_ids_512 = layout_ids_512.squeeze(1).long()
    ratios_generated = layout_onehot_64.mean(dim=(2, 3)).detach()
    ratios_image_device = ratios_generated.to(device=device, dtype=weight_dtype)

    training_config = _load_json(controlnet_root / "training_config.json") or {}
    base_model = args.base_model or training_config.get("base_model")
    if not base_model:
        raise ValueError("base_model is required. Pass --base_model or include it in controlnet training_config.json.")
    prompt = args.prompt or training_config.get("prompt") or "a high-resolution satellite image"
    palette = build_palette(class_names, num_classes, dataset=training_config.get("dataset"))

    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=weight_dtype)
    if controlnet_state_ckpt is not None:
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=num_classes)
        _ensure_identity_class_embedding(controlnet)
        controlnet_state = _load_state_dict(controlnet_state_ckpt / "model.safetensors")
        controlnet.load_state_dict(controlnet_state)
    else:
        controlnet = ControlNetModel.from_pretrained(controlnet_root / "controlnet", torch_dtype=weight_dtype)
        _ensure_identity_class_embedding(controlnet)

    _ensure_identity_class_embedding(unet)
    if args.lora_path is not None:
        unet.load_attn_procs(args.lora_path, weight_name=args.lora_weight_name)
        logger.info("Loaded UNet LoRA from %s (%s)", args.lora_path, args.lora_weight_name)

    time_embed_dim = infer_time_embed_dim_from_config(unet.config.block_out_channels)
    if controlnet_state_ckpt is not None:
        ratio_state_path = controlnet_state_ckpt / "model_1.safetensors"
        ratio_state = _load_state_dict(ratio_state_path)
        ratio_projector = build_ratio_projector_from_state_dict(ratio_state, num_classes=num_classes, embed_dim=time_embed_dim)
        ratio_projector.load_state_dict(ratio_state)
        gate_state_path = controlnet_state_ckpt / "model_2.safetensors"
        gate_state = _load_state_dict(gate_state_path)
    else:
        ratio_projector = _load_ratio_projector(controlnet_root, num_classes, time_embed_dim)
        gate_state_path = controlnet_root / "film_gate.bin"
        gate_state = _load_state_dict(gate_state_path)
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
        raise ValueError(f"Unrecognized FiLM gate checkpoint format: {gate_state_path}")
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
        [prompt] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)

    uncond_inputs = tokenizer(
        [""] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]
    uncond_embeds = uncond_embeds.to(dtype=weight_dtype)

    ratio_emb = ratio_projector(ratios_image_device).to(dtype=weight_dtype)
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
        if init_image.shape[0] != batch_size:
            init_image = init_image.repeat(batch_size, 1, 1, 1)
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
            (batch_size, unet.config.in_channels, args.image_size // 8, args.image_size // 8),
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
            if args.control_scale is not None and float(args.control_scale) != 1.0:
                scale = float(args.control_scale)
                down_samples = tuple(sample * scale for sample in down_samples)
                mid_sample = mid_sample * scale

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

    images = _vae_decode(vae, latents / vae.config.scaling_factor)
    if images.ndim == 3:
        images = images.unsqueeze(0)

    save_dir = Path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    metadata_entries = []
    ratios_layout_list = [float(x) for x in ratios_layout.detach().cpu().tolist()]
    ratios_layout_mask_list = [float(x) for x in ratios_layout_mask.detach().cpu().tolist()]
    ratios_requested_list = [float(x) for x in ratios_requested.detach().cpu().tolist()]
    ratios_known_mask_list = [float(x) for x in ratios_known_mask.detach().cpu().tolist()]

    for idx in range(batch_size):
        suffix = "" if batch_size == 1 else f"_{idx:03d}"
        image_path = save_dir / f"image{suffix}.png"
        layout_path = save_dir / f"layout{suffix}.png"
        layout_color_path = save_dir / f"layout_color{suffix}.png"
        _save_uint8_rgb(images[idx], image_path)
        _save_label_map(layout_ids_512[idx], layout_path)
        layout_color = _colorize_labels(layout_ids_512[idx], palette)
        Image.fromarray(layout_color, mode="RGB").save(layout_color_path)

        metadata = {
            "prompt": prompt,
            "domain": str(args.domain),
            "domain_cond_scale": float(args.domain_cond_scale),
            "ratios": ratios_layout_list,
            "ratios_known_mask": ratios_layout_mask_list,
            "ratios_requested": ratios_requested_list,
            "ratios_requested_mask": ratios_known_mask_list,
            "ratios_generated": [float(x) for x in ratios_generated[idx].detach().cpu().tolist()],
            "class_names": class_names,
            "layout_path": str(layout_path),
            "layout_color_path": str(layout_color_path),
            "image_path": str(image_path),
        }
        if batch_size > 1:
            metadata["sample_index"] = int(idx)
        metadata_entries.append(metadata)

    metadata_path = save_dir / "metadata.json"
    if batch_size == 1:
        metadata_path.write_text(json.dumps(metadata_entries[0], indent=2), encoding="utf-8")
    else:
        metadata_path.write_text(json.dumps(metadata_entries, indent=2), encoding="utf-8")
    logger.info(f"Saved outputs under {args.save_dir}")


if __name__ == "__main__":
    main()
