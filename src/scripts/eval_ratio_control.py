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


def _collect_ratios(dataset, num_samples: int) -> List[torch.Tensor]:
    if dataset is None:
        raise ValueError("--data_root is required for ratio targets.")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    ratios = []
    for batch in loader:
        ratios.append(batch["ratios"][0])
        if len(ratios) >= num_samples:
            break
    if not ratios:
        raise ValueError("No ratios collected from dataset.")
    return ratios


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
    ratio_projector.load_state_dict(torch.load(layout_ckpt / "ratio_projector.bin", map_location="cpu"))
    scheduler = DDPMScheduler.from_pretrained(layout_ckpt / "scheduler")

    dataset = _resolve_dataset(args, num_classes, layout_size)
    ratios_list = _collect_ratios(dataset, args.num_samples)

    layout_unet.to(device)
    ratio_projector.to(device)
    layout_unet.eval()
    ratio_projector.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    errors = []
    for ratios in ratios_list:
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

        layout_ids = latents.argmax(dim=1)[0]
        counts = torch.bincount(layout_ids.view(-1), minlength=num_classes).float()
        r_hat = counts / counts.sum().clamp(min=1.0)
        errors.append((r_hat.cpu() - ratios.cpu()).abs())

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
