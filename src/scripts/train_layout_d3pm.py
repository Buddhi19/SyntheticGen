#!/usr/bin/env python
# coding=utf-8

"""Train a layout D3PM (categorical diffusion) conditioned on class ratios."""

import argparse
import json
import logging
import math
import os
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from tqdm.auto import tqdm

from diffusers import UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

try:
    from .dataset_loveda import GenericSegDataset, LoveDADataset, build_palette, load_class_names
    from .config_utils import apply_config
    from ..models.d3pm import D3PMScheduler
    from ..models.ratio_conditioning import RatioProjector, infer_time_embed_dim_from_config
except ImportError:  # direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.scripts.dataset_loveda import GenericSegDataset, LoveDADataset, build_palette, load_class_names
    from src.scripts.config_utils import apply_config
    from src.models.d3pm import D3PMScheduler
    from src.models.ratio_conditioning import RatioProjector, infer_time_embed_dim_from_config


check_min_version("0.36.0")

logger = get_logger(__name__, log_level="INFO")

if is_wandb_available():
    import wandb  # noqa: F401


def _tv_anisotropic(x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    dx = x[..., 1:, :] - x[..., :-1, :]
    dy = x[..., :, 1:] - x[..., :, :-1]
    if valid_mask is None:
        return dx.abs().mean() + dy.abs().mean()
    vh = valid_mask[..., 1:, :] * valid_mask[..., :-1, :]
    vw = valid_mask[..., :, 1:] * valid_mask[..., :, :-1]
    tv_h = (dx.abs() * vh).sum() / vh.sum().clamp(min=1.0)
    tv_w = (dy.abs() * vw).sum() / vw.sum().clamp(min=1.0)
    return tv_h + tv_w


def _tv_weight(step: int, max_steps: int, lambda_tv: float, warmup_steps: int) -> float:
    if lambda_tv <= 0:
        return 0.0
    if warmup_steps is None or warmup_steps <= 0:
        warmup_steps = max(1, int(0.15 * max_steps))
    if step <= 0:
        return 0.0
    if step >= warmup_steps:
        return float(lambda_tv)
    return float(lambda_tv) * (float(step) / float(warmup_steps))


def _ramp_weight(step: int, max_steps: int, target: float) -> float:
    if target <= 0:
        return 0.0
    warmup = max(1, int(0.15 * max_steps))
    if step <= 0:
        return 0.0
    if step >= warmup:
        return float(target)
    return float(target) * (float(step) / float(warmup))


def _potts_neighbor_disagreement(probs: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    agree_v = (probs[:, :, 1:, :] * probs[:, :, :-1, :]).sum(dim=1)  # (B,H-1,W)
    agree_h = (probs[:, :, :, 1:] * probs[:, :, :, :-1]).sum(dim=1)  # (B,H,W-1)
    dis_v = 1.0 - agree_v
    dis_h = 1.0 - agree_h
    if valid_mask is None:
        return dis_v.mean() + dis_h.mean()
    vh = (valid_mask[..., 1:, :] * valid_mask[..., :-1, :]).squeeze(1)
    vw = (valid_mask[..., :, 1:] * valid_mask[..., :, :-1]).squeeze(1)
    loss_v = (dis_v * vh).sum() / vh.sum().clamp(min=1.0)
    loss_h = (dis_h * vw).sum() / vw.sum().clamp(min=1.0)
    return loss_v + loss_h


def _multiscale_smoothness(
    probs: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = (1, 2, 4, 8)
    tv_total = torch.zeros((), device=probs.device, dtype=probs.dtype)
    potts_total = torch.zeros((), device=probs.device, dtype=probs.dtype)
    count = 0

    for scale in scales:
        if int(scale) == 1:
            p = probs
            v = valid_mask
        else:
            k = int(scale)
            p = F.avg_pool2d(probs, kernel_size=k, stride=k)
            v = None
            if valid_mask is not None:
                v = F.avg_pool2d(valid_mask, kernel_size=k, stride=k)
                v = (v > 0.5).to(dtype=p.dtype)

        tv_total = tv_total + _tv_anisotropic(p, valid_mask=v)
        potts_total = potts_total + _potts_neighbor_disagreement(p, valid_mask=v)
        count += 1

    tv_total = tv_total / float(count)
    potts_total = potts_total / float(count)
    total = 0.25 * tv_total + 0.75 * potts_total
    return total, tv_total, potts_total


def parse_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML/JSON config file; values act as argparse defaults.",
    )
    cfg_args, remaining = base_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Train a layout D3PM conditioned on class ratios.", parents=[base_parser])
    parser.add_argument("--data_root", type=str, default=None, help="Root folder for the dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="generic",
        choices=["loveda", "generic"],
        help="Dataset type to use.",
    )
    parser.add_argument("--output_dir", type=str, default="outputsV2/layout_d3pm")
    parser.add_argument("--image_size", type=int, default=512, help="Dataset image size (square).")
    parser.add_argument("--layout_size", type=int, default=256, help="Layout diffusion size (square).")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes in the dataset.")
    parser.add_argument("--class_names_json", type=str, default=None)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--loveda_split", type=str, default="Train")
    parser.add_argument("--loveda_domains", type=str, default="Urban,Rural")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel width for the layout UNet.")
    parser.add_argument("--layers_per_block", type=int, default=1, help="Number of layers per UNet block.")
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=200000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lambda_ratio", type=float, default=1.0)
    parser.add_argument(
        "--ratio_conditioning",
        type=str,
        default="full",
        choices=["full", "masked"],
        help="Conditioning type for ratios. Use 'masked' to train with partial ratio constraints.",
    )
    parser.add_argument(
        "--ratio_known_weight",
        type=float,
        default=1.0,
        help="When --ratio_conditioning=masked, weight for known ratios in ratio loss (higher = enforce known more).",
    )
    parser.add_argument(
        "--ratio_unknown_weight",
        type=float,
        default=0.0,
        help="When --ratio_conditioning=masked, weight for unknown ratios in ratio loss (0 disables).",
    )
    parser.add_argument(
        "--p_keep",
        type=float,
        default=1.0,
        help="When --ratio_conditioning=masked, probability each class ratio is revealed during training.",
    )
    parser.add_argument(
        "--known_count_min",
        type=int,
        default=1,
        help="When --ratio_conditioning=masked, enforce at least this many known classes per sample.",
    )
    parser.add_argument(
        "--known_count_max",
        type=int,
        default=0,
        help="When --ratio_conditioning=masked, if >0 enforce at most this many known classes per sample.",
    )
    parser.add_argument(
        "--present_eps",
        type=float,
        default=0.001,
        help="When --ratio_conditioning=masked, classes with ratio_true > present_eps are considered 'present' "
        "for known-mask sampling.",
    )
    parser.add_argument(
        "--add_negative_class",
        action="store_true",
        help="When --ratio_conditioning=masked, also reveal one absent class (ratio==0) as a 'known negative' "
        "to stabilize conditioning.",
    )
    parser.add_argument(
        "--min_unknown_for_prior",
        type=int,
        default=3,
        help="Compute prior KL only if unknown class count >= this value (masked mode).",
    )
    parser.add_argument(
        "--min_unknown_mass_for_prior",
        type=float,
        default=0.05,
        help="Compute prior KL only if total unknown mass >= this value (masked mode).",
    )
    parser.add_argument(
        "--lambda_ce",
        type=float,
        default=0.0,
        help="Auxiliary CE on x0 logits vs ground-truth labels (D3PM recommendation; can stabilize training).",
    )
    parser.add_argument("--lambda_prior", type=float, default=0.0, help="Optional prior loss weight on unknown ratios.")
    parser.add_argument(
        "--ratio_prior_json",
        type=str,
        default=None,
        help="Optional JSON file with a global class-ratio prior (list of K floats).",
    )
    parser.add_argument("--ratio_temp", type=float, default=1.0)
    parser.add_argument(
        "--ratio_pool",
        type=int,
        default=8,
        help="Average-pool kernel/stride used before computing ratio loss. 1 disables.",
    )
    parser.add_argument("--lambda_tv", type=float, default=0.0)
    parser.add_argument("--tv_warmup_steps", type=int, default=-1)
    parser.add_argument("--lambda_smooth", type=float, default=0.0)
    parser.add_argument("--smooth_warmup_steps", type=int, default=-1)
    parser.add_argument("--lambda_ent", type=float, default=0.0)
    parser.add_argument("--ent_warmup_steps", type=int, default=-1)
    parser.add_argument("--viz_smooth", type=int, default=0)
    parser.add_argument("--sample_num_inference_steps", type=int, default=50)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="The integration to report to (tensorboard, wandb, or none).",
    )
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    if cfg_args.config:
        apply_config(parser, cfg_args.config)
    args = parser.parse_args(remaining)
    args.config = cfg_args.config

    if args.data_root is None:
        parser.error("--data_root is required (pass it directly or via --config).")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def _resolve_dataset(args, num_classes):
    if args.dataset == "loveda":
        domains = [domain.strip() for domain in args.loveda_domains.split(",") if domain.strip()]
        return LoveDADataset(
            args.data_root,
            image_size=args.image_size,
            split=args.loveda_split,
            domains=domains,
            ignore_index=args.ignore_index,
            num_classes=num_classes,
            return_layouts=True,
            layout_size=args.layout_size,
        )
    return GenericSegDataset(
        args.data_root,
        image_size=args.image_size,
        num_classes=num_classes,
        ignore_index=None,
        return_layouts=True,
        layout_size=args.layout_size,
    )


def _get_tb_writer(accelerator: Accelerator):
    try:
        tracker = accelerator.get_tracker("tensorboard")
    except Exception:
        return None
    if tracker is None:
        return None
    return getattr(tracker, "writer", None)


def _colorize_labels(label_map: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    labels = label_map.detach().cpu().numpy().astype(np.int64)
    return palette[labels]


def _log_layout_sample(
    accelerator: Accelerator,
    unet: UNet2DModel,
    ratio_projector: RatioProjector,
    scheduler: D3PMScheduler,
    ratios: torch.Tensor,
    known_mask: torch.Tensor | None,
    palette: np.ndarray,
    layout_size: int,
    num_inference_steps: int,
    output_dir: str,
    step: int,
    seed: int,
    ratio_pool: int,
    ratio_temp: float,
    viz_smooth: int,
) -> None:
    writer = _get_tb_writer(accelerator)
    generator = torch.Generator(device=ratios.device).manual_seed(int(seed))

    if getattr(ratio_projector, "input_dim", ratios.shape[0]) != ratios.shape[0]:
        if known_mask is None:
            known_mask = torch.ones_like(ratios)
        ratio_emb = ratio_projector(ratios.unsqueeze(0), known_mask.unsqueeze(0))
    else:
        ratio_emb = ratio_projector(ratios.unsqueeze(0))

    scheduler.set_timesteps(int(num_inference_steps), device=ratios.device)
    x_t = torch.randint(
        0,
        int(palette.shape[0]),
        size=(1, layout_size, layout_size),
        device=ratios.device,
        dtype=torch.long,
        generator=generator,
    )

    logits_last = None
    with torch.no_grad():
        timesteps = scheduler.timesteps
        assert timesteps is not None
        for idx, t in enumerate(timesteps):
            t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            t_prev = int(timesteps[idx + 1].item()) if idx + 1 < int(timesteps.numel()) else 0
            x_t_onehot = F.one_hot(x_t, num_classes=int(palette.shape[0])).permute(0, 3, 1, 2).to(
                device=ratios.device, dtype=ratio_emb.dtype
            )
            logits_x0 = unet(x_t_onehot, torch.tensor([t_int], device=ratios.device), class_labels=ratio_emb).sample
            logits_last = logits_x0
            logp_prev = scheduler.p_theta_posterior_logprobs(
                x_t, t=torch.tensor([t_int], device=ratios.device), logits_x0=logits_x0, t_prev=torch.tensor([t_prev], device=ratios.device)
            )
            x_t = scheduler.sample_from_logprobs(logp_prev, generator=generator)

    layout_ids = x_t[0]
    color = _colorize_labels(layout_ids, palette)
    samples_dir = Path(output_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_path = samples_dir / f"layout_step_{step:06d}.png"
    Image.fromarray(color, mode="RGB").save(sample_path)

    ratios_target = ratios.to(dtype=torch.float32)
    layout_onehot = F.one_hot(layout_ids, num_classes=int(palette.shape[0])).permute(2, 0, 1).float()
    r_hat = layout_onehot.mean(dim=(1, 2))
    if known_mask is None:
        ratio_mae = (r_hat - ratios_target).abs().mean()
    else:
        mask = known_mask.to(dtype=r_hat.dtype)
        ratio_mae = ((r_hat - ratios_target).abs() * mask).sum() / mask.sum().clamp(min=1.0)

    if viz_smooth is not None and int(viz_smooth) > 1 and logits_last is not None:
        k = int(viz_smooth)
        pad = k // 2
        logits_last = F.avg_pool2d(logits_last, kernel_size=k, stride=1, padding=pad)

    if writer is not None:
        writer.add_image("samples/layout", color, step, dataformats="HWC")
        writer.add_scalar("samples/ratio_mae", float(ratio_mae.item()), step)


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    log_with = None if args.report_to == "none" else args.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.seed is not None:
        set_seed(args.seed)

    class_names, num_classes = load_class_names(args.class_names_json, args.num_classes, args.dataset)
    args.num_classes = num_classes
    palette = build_palette(class_names, num_classes, dataset=args.dataset)

    if args.ratio_conditioning == "masked":
        if int(args.known_count_min) < 1 or int(args.known_count_min) > num_classes:
            raise ValueError(f"--known_count_min must be in [1, {num_classes}].")
        if int(args.known_count_max) > 0 and int(args.known_count_max) < int(args.known_count_min):
            raise ValueError("--known_count_max must be 0 or >= --known_count_min.")
        if float(args.ratio_known_weight) < 0 or float(args.ratio_unknown_weight) < 0:
            raise ValueError("--ratio_known_weight/--ratio_unknown_weight must be >= 0.")

    ratio_prior = None
    if args.ratio_prior_json:
        prior_path = Path(args.ratio_prior_json)
        data = json.loads(prior_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) != num_classes:
            raise ValueError(f"ratio_prior_json must be a list of {num_classes} floats.")
        ratio_prior = torch.tensor([float(x) for x in data], dtype=torch.float32)

    train_dataset = _resolve_dataset(args, num_classes)

    def collate_fn(examples):
        label_key = f"label_{args.layout_size}"
        valid_key = f"valid_{args.layout_size}"
        ratios_key = f"ratios_{args.layout_size}"
        labels = torch.stack([ex[label_key] for ex in examples])
        ratios = torch.stack([ex.get(ratios_key, ex["ratios"]) for ex in examples])
        valids = torch.stack([ex.get(valid_key, ex["valid_64"]) for ex in examples])
        return {"labels": labels, "ratios": ratios, "valids": valids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    block_out_channels = (
        args.base_channels,
        args.base_channels * 2,
        args.base_channels * 2,
        args.base_channels * 2,
    )
    unet = UNet2DModel(
        sample_size=args.layout_size,
        in_channels=num_classes,
        out_channels=num_classes,
        layers_per_block=args.layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        class_embed_type="identity",
    )

    time_embed_dim = infer_time_embed_dim_from_config(unet.config.block_out_channels)
    if args.ratio_conditioning == "masked":
        ratio_projector = RatioProjector(
            num_classes,
            time_embed_dim,
            input_dim=2 * num_classes + 1,
            layer_norm=True,
        )
    else:
        ratio_projector = RatioProjector(num_classes, time_embed_dim)

    scheduler = D3PMScheduler(
        num_classes=num_classes,
        num_timesteps=int(args.num_train_timesteps),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
        device="cpu",
    )

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size

    optimizer = torch.optim.AdamW(
        chain(unet.parameters(), ratio_projector.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    unet, ratio_projector, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, ratio_projector, optimizer, train_dataloader, lr_scheduler
    )
    scheduler.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        if Path(args.resume_from_checkpoint).name.startswith("checkpoint-"):
            global_step = int(Path(args.resume_from_checkpoint).name.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def _sample_known_mask_present_aware(ratios_true: torch.Tensor) -> torch.Tensor:
        if ratios_true.ndim != 2:
            raise ValueError(f"Expected ratios_true (B,C), got {tuple(ratios_true.shape)}")
        batch_size, ncls = ratios_true.shape
        device = ratios_true.device
        dtype = ratios_true.dtype

        p_keep = float(args.p_keep)
        known_min = int(args.known_count_min)
        known_max = int(args.known_count_max)
        eps = float(args.present_eps)

        present = ratios_true.float() > eps  # (B,C) bool

        if p_keep <= 0.0:
            mask = torch.zeros((batch_size, ncls), device=device, dtype=dtype)
        elif p_keep >= 1.0:
            mask = present.to(dtype=dtype)
        else:
            rnd = torch.rand((batch_size, ncls), device=device)
            mask = ((rnd < p_keep) & present).to(dtype=dtype)

        if known_max > 0 and known_max < ncls:
            for i in range(batch_size):
                on = torch.nonzero(mask[i] > 0, as_tuple=False).flatten()
                if on.numel() <= known_max:
                    continue
                keep_idx = on[torch.randperm(on.numel(), device=device)[:known_max]]
                new_mask = torch.zeros((ncls,), device=device, dtype=dtype)
                new_mask[keep_idx] = 1.0
                mask[i] = new_mask

        for i in range(batch_size):
            on = torch.nonzero(mask[i] > 0, as_tuple=False).flatten()
            if on.numel() >= known_min:
                continue
            candidates = torch.nonzero(present[i], as_tuple=False).flatten()
            if candidates.numel() == 0:
                candidates = torch.arange(ncls, device=device)
            perm = candidates[torch.randperm(candidates.numel(), device=device)]
            need = known_min - on.numel()
            mask[i, perm[:need]] = 1.0

        if bool(args.add_negative_class):
            for i in range(batch_size):
                absent = torch.nonzero(~present[i], as_tuple=False).flatten()
                if absent.numel() == 0:
                    continue
                j = absent[torch.randint(0, absent.numel(), (1,), device=device).item()]
                mask[i, j] = 1.0
        return mask

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        ratio_projector.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                labels = batch["labels"].to(device=accelerator.device, dtype=torch.long)  # (B,H,W)
                ratios_true = batch["ratios"].to(device=accelerator.device, dtype=torch.float32)  # (B,K)
                valid_mask = batch["valids"].to(device=accelerator.device, dtype=torch.float32)  # (B,1,H,W)
                valid_px = valid_mask.squeeze(1) > 0.5  # (B,H,W)

                x0 = labels
                x0_safe = x0.clone()
                x0_safe[~valid_px] = 0

                bsz = x0_safe.shape[0]
                timesteps = torch.randint(1, int(args.num_train_timesteps) + 1, (bsz,), device=accelerator.device).long()
                x_t = scheduler.q_sample(x0_safe, timesteps)  # (B,H,W)
                x_t_onehot = F.one_hot(x_t, num_classes=num_classes).permute(0, 3, 1, 2).float()
                x_t_onehot = (x_t_onehot * valid_mask).to(dtype=weight_dtype)

                known_mask = None
                ratios_obs = ratios_true
                if args.ratio_conditioning == "masked":
                    known_mask = _sample_known_mask_present_aware(ratios_true)
                    ratios_obs = ratios_true * known_mask
                    ratio_emb = ratio_projector(ratios_obs, known_mask)
                else:
                    ratio_emb = ratio_projector(ratios_true)

                logits_x0 = unet(x_t_onehot, timesteps, class_labels=ratio_emb).sample  # (B,K,H,W)

                y = x0_safe.clone()
                y = y.masked_fill(~valid_px, -100)
                ce_map = F.cross_entropy(logits_x0.float(), y.long(), ignore_index=-100, reduction="none")  # (B,H,W)
                v = valid_mask.squeeze(1).float()
                denom = v.sum(dim=(1, 2)).clamp(min=1.0)
                ce_per = (ce_map * v).sum(dim=(1, 2)) / denom
                ce_aux = ce_per.mean()

                vlb_per = torch.zeros_like(ce_per)
                t_is1 = timesteps == 1
                t_gt1 = timesteps > 1
                if torch.any(t_is1):
                    vlb_per[t_is1] = ce_per[t_is1]
                if torch.any(t_gt1):
                    x_t_m = x_t[t_gt1]
                    x0_m = x0_safe[t_gt1]
                    t_m = timesteps[t_gt1]
                    t_prev_m = t_m - 1
                    logits_m = logits_x0[t_gt1]
                    q_logp = scheduler.q_posterior_logprobs(x_t_m, x0_m, t=t_m, t_prev=t_prev_m)
                    p_logp = scheduler.p_theta_posterior_logprobs(x_t_m, t=t_m, logits_x0=logits_m, t_prev=t_prev_m)
                    kl_map = (torch.exp(q_logp) * (q_logp - p_logp)).sum(dim=-1)  # (Bm,H,W)
                    v_m = v[t_gt1]
                    denom_m = v_m.sum(dim=(1, 2)).clamp(min=1.0)
                    kl_per = (kl_map * v_m).sum(dim=(1, 2)) / denom_m
                    vlb_per[t_gt1] = kl_per
                vlb_loss = vlb_per.mean()

                probs = torch.softmax(logits_x0 / float(args.ratio_temp), dim=1)

                ent_loss = torch.tensor(0.0, device=probs.device)
                ent_w = 0.0
                if args.lambda_ent is not None and float(args.lambda_ent) > 0:
                    p = probs.float().clamp(min=1e-8)
                    ent = -(p * p.log()).sum(dim=1)  # (B,H,W)
                    ent_loss = (ent * v).sum() / v.sum().clamp(min=1.0)
                    ent_w = _tv_weight(global_step, args.max_train_steps, float(args.lambda_ent), int(args.ent_warmup_steps))

                smooth_loss = torch.tensor(0.0, device=probs.device)
                smooth_tv = torch.tensor(0.0, device=probs.device)
                smooth_potts = torch.tensor(0.0, device=probs.device)
                smooth_w = 0.0
                if args.lambda_smooth is not None and float(args.lambda_smooth) > 0:
                    probs_f = probs.float()
                    valid_f = valid_mask.float() if valid_mask is not None else None
                    smooth_loss_f, smooth_tv_f, smooth_potts_f = _multiscale_smoothness(probs_f, valid_mask=valid_f)
                    smooth_loss = smooth_loss_f.to(dtype=probs.dtype)
                    smooth_tv = smooth_tv_f.to(dtype=probs.dtype)
                    smooth_potts = smooth_potts_f.to(dtype=probs.dtype)
                    smooth_w = _tv_weight(
                        global_step, args.max_train_steps, float(args.lambda_smooth), int(args.smooth_warmup_steps)
                    )

                if args.ratio_pool is not None and int(args.ratio_pool) > 1:
                    k = int(args.ratio_pool)
                    probs_p = F.avg_pool2d(probs, kernel_size=k, stride=k)
                    valid_p = F.avg_pool2d(valid_mask, kernel_size=k, stride=k)
                else:
                    probs_p = probs
                    valid_p = valid_mask

                weighted = (probs_p * valid_p).sum(dim=(2, 3))
                denom_p = valid_p.sum(dim=(2, 3)).clamp(min=1.0)
                r_hat = weighted / denom_p

                if known_mask is None:
                    ratio_loss = F.mse_loss(r_hat.float(), ratios_true.float(), reduction="mean")
                    ratio_loss_known = ratio_loss.detach()
                    ratio_loss_unknown = torch.tensor(0.0, device=r_hat.device)
                    prior_loss = torch.tensor(0.0, device=r_hat.device)
                else:
                    mask_f = known_mask.to(dtype=r_hat.dtype)
                    unknown_f = (1.0 - mask_f).clamp(min=0.0, max=1.0)
                    diff2 = (r_hat - ratios_true).float().pow(2)
                    ratio_loss_known = (diff2 * mask_f.float()).sum() / mask_f.sum().clamp(min=1.0)
                    ratio_loss_unknown = (diff2 * unknown_f.float()).sum() / unknown_f.sum().clamp(min=1.0)
                    ratio_loss = float(args.ratio_known_weight) * ratio_loss_known + float(args.ratio_unknown_weight) * ratio_loss_unknown
                    prior_loss = torch.tensor(0.0, device=r_hat.device)
                    if ratio_prior is not None and args.lambda_prior is not None and float(args.lambda_prior) > 0:
                        unknown_count = (unknown_f > 0.5).sum(dim=1)
                        unknown_mass = (r_hat.float() * unknown_f.float()).sum(dim=1)
                        eligible = (unknown_count >= int(args.min_unknown_for_prior)) & (
                            unknown_mass >= float(args.min_unknown_mass_for_prior)
                        )
                        if torch.any(eligible):
                            r_hat_rem = r_hat.float() * unknown_f.float()
                            r_hat_rem = r_hat_rem / r_hat_rem.sum(dim=1, keepdim=True).clamp(min=1e-8)
                            p0 = ratio_prior.to(device=r_hat.device, dtype=torch.float32).unsqueeze(0) * unknown_f.float()
                            p0 = p0 / p0.sum(dim=1, keepdim=True).clamp(min=1e-8)
                            kl_vec = (r_hat_rem * ((r_hat_rem + 1e-8).log() - (p0 + 1e-8).log())).sum(dim=1)
                            prior_loss = kl_vec[eligible].mean()
                        else:
                            prior_loss = torch.tensor(0.0, device=r_hat.device)

                tv_loss = torch.tensor(0.0, device=probs.device)
                tv_w = 0.0
                if args.lambda_tv is not None and float(args.lambda_tv) > 0:
                    tv_loss = _tv_anisotropic(probs, valid_mask=valid_mask)
                    tv_w = _tv_weight(global_step, args.max_train_steps, float(args.lambda_tv), int(args.tv_warmup_steps))

                ce_w = _ramp_weight(global_step, args.max_train_steps, float(args.lambda_ce))
                loss = (
                    vlb_loss
                    + ce_w * ce_aux
                    + args.lambda_ratio * ratio_loss
                    + float(args.lambda_prior) * prior_loss
                    + tv_w * tv_loss
                    + smooth_w * smooth_loss
                    + ent_w * ent_loss
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(chain(unet.parameters(), ratio_projector.parameters()), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.logging_steps == 0:
                    accelerator.log(
                        {
                            "train_loss": loss.detach().item(),
                            "vlb_loss": vlb_loss.detach().item(),
                            "ce_aux": ce_aux.detach().item(),
                            "ce_w": float(ce_w),
                            "ratio_loss": ratio_loss.detach().item(),
                            "ratio_loss_known": ratio_loss_known.detach().item(),
                            "ratio_loss_unknown": ratio_loss_unknown.detach().item(),
                            "ratio_known_w": float(args.ratio_known_weight),
                            "ratio_unknown_w": float(args.ratio_unknown_weight),
                            "prior_loss": prior_loss.detach().item(),
                            "tv_loss": tv_loss.detach().item(),
                            "tv_w": float(tv_w),
                            "smooth_loss": smooth_loss.detach().item(),
                            "smooth_tv": smooth_tv.detach().item(),
                            "smooth_potts": smooth_potts.detach().item(),
                            "smooth_w": float(smooth_w),
                            "ent_loss": ent_loss.detach().item(),
                            "ent_w": float(ent_w),
                        },
                        step=global_step,
                    )

                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        ratios_sample = ratios_true[0].detach()
                        known_mask_sample = known_mask[0].detach() if known_mask is not None else None
                        was_training = unet.training
                        unet.eval()
                        ratio_projector.eval()
                        _log_layout_sample(
                            accelerator=accelerator,
                            unet=accelerator.unwrap_model(unet),
                            ratio_projector=accelerator.unwrap_model(ratio_projector),
                            scheduler=scheduler,
                            ratios=ratios_sample,
                            known_mask=known_mask_sample,
                            palette=palette,
                            layout_size=args.layout_size,
                            num_inference_steps=args.sample_num_inference_steps,
                            output_dir=args.output_dir,
                            step=global_step,
                            seed=args.sample_seed,
                            ratio_pool=args.ratio_pool,
                            ratio_temp=args.ratio_temp,
                            viz_smooth=args.viz_smooth,
                        )
                        if was_training:
                            unet.train()
                            ratio_projector.train()

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_ratio = accelerator.unwrap_model(ratio_projector)
        unwrapped_unet.save_pretrained(os.path.join(args.output_dir, "layout_unet"))
        torch.save(unwrapped_ratio.state_dict(), os.path.join(args.output_dir, "ratio_projector.bin"))
        scheduler.save_config(os.path.join(args.output_dir, "d3pm_config.json"))
        with open(os.path.join(args.output_dir, "class_names.json"), "w", encoding="utf-8") as handle:
            json.dump(class_names, handle, indent=2)
        with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as handle:
            tv_warmup_steps = int(args.tv_warmup_steps)
            if tv_warmup_steps <= 0:
                tv_warmup_steps = max(1, int(0.15 * int(args.max_train_steps)))
            smooth_warmup_steps = int(args.smooth_warmup_steps)
            if smooth_warmup_steps <= 0:
                smooth_warmup_steps = max(1, int(0.15 * int(args.max_train_steps)))
            ent_warmup_steps = int(args.ent_warmup_steps)
            if ent_warmup_steps <= 0:
                ent_warmup_steps = max(1, int(0.15 * int(args.max_train_steps)))
            json.dump(
                {
                    "diffusion_type": "d3pm",
                    "dataset": args.dataset,
                    "num_classes": num_classes,
                    "layout_size": args.layout_size,
                    "image_size": args.image_size,
                    "ignore_index": args.ignore_index,
                    "num_train_timesteps": args.num_train_timesteps,
                    "beta_start": float(args.beta_start),
                    "beta_end": float(args.beta_end),
                    "base_channels": args.base_channels,
                    "layers_per_block": args.layers_per_block,
                    "lambda_ratio": args.lambda_ratio,
                    "ratio_conditioning": args.ratio_conditioning,
                    "p_keep": args.p_keep,
                    "known_count_min": args.known_count_min,
                    "known_count_max": args.known_count_max,
                    "present_eps": args.present_eps,
                    "add_negative_class": bool(args.add_negative_class),
                    "min_unknown_for_prior": args.min_unknown_for_prior,
                    "min_unknown_mass_for_prior": args.min_unknown_mass_for_prior,
                    "ratio_known_weight": args.ratio_known_weight,
                    "ratio_unknown_weight": args.ratio_unknown_weight,
                    "lambda_prior": args.lambda_prior,
                    "ratio_prior_json": args.ratio_prior_json,
                    "ratio_projector_input_dim": getattr(unwrapped_ratio, "input_dim", num_classes),
                    "ratio_temp": args.ratio_temp,
                    "ratio_pool": args.ratio_pool,
                    "lambda_ce": args.lambda_ce,
                    "lambda_tv": args.lambda_tv,
                    "tv_warmup_steps": tv_warmup_steps,
                    "lambda_smooth": args.lambda_smooth,
                    "smooth_warmup_steps": smooth_warmup_steps,
                    "lambda_ent": args.lambda_ent,
                    "ent_warmup_steps": ent_warmup_steps,
                    "viz_smooth": args.viz_smooth,
                },
                handle,
                indent=2,
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()

