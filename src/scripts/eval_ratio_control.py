#!/usr/bin/env python
# coding=utf-8

"""Evaluate ratio control for the layout DDPM."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import DDPMScheduler, UNet2DModel

try:
    from .dataset_loveda import GenericSegDataset, LoveDADataset, load_class_names
    from ..models.ratio_conditioning import RatioProjector, infer_time_embed_dim_from_config
except ImportError:  # direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.scripts.dataset_loveda import GenericSegDataset, LoveDADataset, load_class_names
    from src.models.ratio_conditioning import RatioProjector, infer_time_embed_dim_from_config


logger = logging.getLogger(__name__)


def _safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ratio control for layout generation.")
    parser.add_argument("--layout_ckpt", type=str, required=True, help="Layout DDPM checkpoint directory.")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root for ratio targets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="generic",
        choices=["loveda", "generic"],
        help="Dataset type for ratio targets.",
    )
    parser.add_argument("--class_names_json", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--loveda_split", type=str, default="Val")
    parser.add_argument("--loveda_domains", type=str, default="Urban,Rural")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--ratio_pool",
        type=int,
        default=8,
        help="Average-pool kernel/stride before computing ratios. Must match training.",
    )
    parser.add_argument(
        "--ratio_temp",
        type=float,
        default=1.0,
        help="Temperature for softmax when computing soft ratios (should match training).",
    )
    parser.add_argument(
        "--use_soft_ratios",
        action="store_true",
        help="If set, compute ratios from softmax probs (preferred). Else use argmax hard map.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", type=str, default="outputsV2/eval_ratio_control.json")
    parser.add_argument("--errors_path", type=str, default=None, help="Optional JSON path for per-sample errors.")
    return parser.parse_args()


def _resolve_dataset(args, num_classes: int, layout_size: int):
    if args.data_root is None:
        return None
    if args.dataset == "loveda":
        domains = [domain.strip() for domain in args.loveda_domains.split(",") if domain.strip()]
        return LoveDADataset(
            args.data_root,
            image_size=layout_size * 8,
            split=args.loveda_split,
            domains=domains,
            ignore_index=args.ignore_index,
            num_classes=num_classes,
            return_layouts=True,
            layout_size=layout_size,
        )
    return GenericSegDataset(
        args.data_root,
        image_size=layout_size * 8,
        num_classes=num_classes,
        ignore_index=args.ignore_index,
        return_layouts=True,
        layout_size=layout_size,
    )


def _collect_ratio_targets(dataset, num_samples: int, layout_size: int):
    if dataset is None:
        raise ValueError("--data_root is required for ratio targets.")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    targets = []
    for batch in loader:
        ratios = batch["ratios"][0]
        valid = batch.get("valid_64")
        if valid is None:
            valid = torch.ones((1, layout_size, layout_size), dtype=torch.float32)
        else:
            valid = valid[0].float()
        targets.append((ratios, valid))
        if len(targets) >= num_samples:
            break
    if not targets:
        raise ValueError("No ratios collected from dataset.")
    return targets


def compute_pooled_ratios_from_logits(
    x_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    ratio_pool: int = 8,
    ratio_temp: float = 1.0,
    use_soft: bool = True,
) -> torch.Tensor:
    """
    x_logits: (B,K,H,W) logits (predicted x0 in logit space)
    valid_mask: (B,1,H,W) float mask in {0,1}
    returns r_hat: (B,K)
    """
    if use_soft:
        probs = torch.softmax(x_logits / float(ratio_temp), dim=1)
    else:
        ids = x_logits.argmax(dim=1)
        probs = F.one_hot(ids, num_classes=x_logits.shape[1]).permute(0, 3, 1, 2).float()

    k = int(ratio_pool) if ratio_pool is not None else 1
    if k > 1:
        probs = F.avg_pool2d(probs, kernel_size=k, stride=k)
        valid_mask = F.avg_pool2d(valid_mask, kernel_size=k, stride=k)

    weighted = (probs * valid_mask).sum(dim=(2, 3))
    denom = valid_mask.sum(dim=(2, 3)).clamp(min=1.0)
    return weighted / denom


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    layout_ckpt = Path(args.layout_ckpt)
    if not layout_ckpt.exists():
        raise FileNotFoundError(f"Layout checkpoint not found: {layout_ckpt}")

    layout_unet = UNet2DModel.from_pretrained(layout_ckpt / "layout_unet")
    num_classes = layout_unet.config.in_channels
    class_names, _ = load_class_names(args.class_names_json, args.num_classes or num_classes, args.dataset)
    layout_size = layout_unet.config.sample_size
    time_embed_dim = infer_time_embed_dim_from_config(layout_unet.config.block_out_channels)
    ratio_projector = RatioProjector(num_classes, time_embed_dim)
    ratio_projector.load_state_dict(_safe_torch_load(layout_ckpt / "ratio_projector.bin", map_location="cpu"))
    scheduler = DDPMScheduler.from_pretrained(layout_ckpt / "scheduler")

    dataset = _resolve_dataset(args, num_classes, layout_size)
    targets = _collect_ratio_targets(dataset, args.num_samples, layout_size)

    layout_unet.to(device)
    ratio_projector.to(device)
    layout_unet.eval()
    ratio_projector.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    errors = []
    for ratios, valid in targets:
        ratios = ratios.to(device=device, dtype=torch.float32)
        ratio_emb = ratio_projector(ratios.unsqueeze(0))
        latents = torch.randn(
            (1, num_classes, layout_size, layout_size),
            generator=generator,
            device=device,
            dtype=ratio_emb.dtype,
        )
        latents = latents * scheduler.init_noise_sigma
        with torch.no_grad():
            for timestep in scheduler.timesteps:
                noise_pred = layout_unet(latents, timestep, class_labels=ratio_emb).sample
                latents = scheduler.step(noise_pred, timestep, latents).prev_sample

        valid_mask = valid.to(device=device, dtype=torch.float32).unsqueeze(0)
        r_hat = compute_pooled_ratios_from_logits(
            x_logits=latents,
            valid_mask=valid_mask,
            ratio_pool=args.ratio_pool,
            ratio_temp=args.ratio_temp,
            use_soft=args.use_soft_ratios,
        )[0]
        errors.append((r_hat.detach().cpu() - ratios.detach().cpu()).abs())

    error_tensor = torch.stack(errors, dim=0)
    mean_abs_error_per_class = error_tensor.mean(dim=0)
    mean_abs_error = mean_abs_error_per_class.mean().item()
    worst_class_error = mean_abs_error_per_class.max().item()

    results = {
        "num_samples": len(errors),
        "mean_abs_error": mean_abs_error,
        "worst_class_error": worst_class_error,
        "mean_abs_error_per_class": mean_abs_error_per_class.tolist(),
        "class_names": class_names,
        "ratio_pool": int(args.ratio_pool),
        "ratio_temp": float(args.ratio_temp),
        "use_soft_ratios": bool(args.use_soft_ratios),
    }

    os.makedirs(Path(args.output_path).parent, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.errors_path is not None:
        os.makedirs(Path(args.errors_path).parent, exist_ok=True)
        error_list = [err.tolist() for err in errors]
        Path(args.errors_path).write_text(json.dumps(error_list, indent=2), encoding="utf-8")
    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
