#!/usr/bin/env python
# coding=utf-8

"""Train ControlNet with layout conditioning and ratio-based FiLM gating."""

import argparse
import json
import logging
import math
import os
from itertools import chain
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

try:
    from .dataset_loveda import GenericSegDataset, LoveDADataset, build_palette, load_class_names
    from ..models.ratio_conditioning import (
        PerChannelResidualFiLMGate,
        RatioProjector,
        infer_time_embed_dim_from_config,
    )
except ImportError:  # direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.scripts.dataset_loveda import GenericSegDataset, LoveDADataset, build_palette, load_class_names
    from src.models.ratio_conditioning import (
        PerChannelResidualFiLMGate,
        RatioProjector,
        infer_time_embed_dim_from_config,
    )


check_min_version("0.36.0")

logger = get_logger(__name__, log_level="INFO")

if is_wandb_available():
    import wandb  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Train ControlNet with layout + ratio conditioning.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--output_dir", type=str, default="outputsV2/controlnet_ratio")
    parser.add_argument("--data_root", type=str, required=True, help="Root folder for the dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="generic",
        choices=["loveda", "generic"],
        help="Dataset type to use.",
    )
    parser.add_argument("--image_size", type=int, default=512, help="Training image size (square).")
    parser.add_argument("--layout_size", type=int, default=256, help="Coarse layout size used for mask mixing.")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes in the dataset.")
    parser.add_argument("--class_names_json", type=str, default=None)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--loveda_split", type=str, default="Train")
    parser.add_argument("--loveda_domains", type=str, default="Urban,Rural")
    parser.add_argument("--prompt", type=str, default="a high-resolution satellite image")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
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
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument(
        "--mask_mix_prob",
        type=float,
        default=0.5,
        help="Probability of using upsampled-from-64 masks instead of true 512 masks.",
    )
    parser.add_argument(
        "--cfg_dropout_prob",
        type=float,
        default=0.1,
        help="Probability to drop text conditioning during training (CFG dropout).",
    )
    parser.add_argument(
        "--cond_dropout_prob",
        type=float,
        default=0.05,
        help="Probability to drop control + ratio conditioning during training.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="Min-SNR gamma. Set <=0 to disable.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Use EMA for ControlNet + ratio modules.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--noise_offset", type=float, default=0.0, help="Optional offset noise strength (0 disables).")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--sample_num_inference_steps", type=int, default=30)
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
    args = parser.parse_args()

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


def _infer_num_down_residuals(controlnet) -> int:
    if hasattr(controlnet, "controlnet_down_blocks"):
        return len(controlnet.controlnet_down_blocks)
    if hasattr(controlnet, "config") and hasattr(controlnet.config, "down_block_types"):
        layers = getattr(controlnet.config, "layers_per_block", 1)
        return 1 + len(controlnet.config.down_block_types) * int(layers)
    return len(getattr(controlnet, "down_blocks", []))


def _ensure_identity_class_embedding(model):
    model.register_to_config(class_embed_type="identity", num_class_embeds=None, projection_class_embeddings_input_dim=None)
    model.class_embedding = nn.Identity()


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


def _vae_decode(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latents)
    if isinstance(decoded, torch.Tensor):
        return decoded
    if hasattr(decoded, "sample"):
        return decoded.sample
    if isinstance(decoded, (tuple, list)) and decoded:
        return decoded[0]
    raise TypeError(f"Unexpected VAE decode output type: {type(decoded)}")


def _save_uint8_rgb(image: torch.Tensor, save_path: Path) -> np.ndarray:
    image = image.detach().cpu()
    image = (image / 2.0 + 0.5).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(save_path)
    return array


def _log_controlnet_sample(
    accelerator: Accelerator,
    controlnet: ControlNetModel,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    prompt_embeds: torch.Tensor,
    ratio_projector: RatioProjector,
    film_gate: nn.Module,
    scheduler: DDIMScheduler,
    layouts: torch.Tensor,
    ratios: torch.Tensor,
    output_dir: str,
    step: int,
    seed: int,
    palette: np.ndarray,
    num_inference_steps: int,
) -> None:
    writer = _get_tb_writer(accelerator)
    generator = torch.Generator(device=layouts.device).manual_seed(seed)

    ratio_emb = ratio_projector(ratios.unsqueeze(0)).to(dtype=prompt_embeds.dtype)
    latents = torch.randn(
        (1, unet.config.in_channels, layouts.shape[-1] // 8, layouts.shape[-1] // 8),
        generator=generator,
        device=layouts.device,
        dtype=prompt_embeds.dtype,
    )
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps, device=layouts.device)

    with torch.no_grad():
        for timestep in scheduler.timesteps:
            down_samples, mid_sample = controlnet(
                latents,
                timestep,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=layouts,
                class_labels=ratio_emb,
                return_dict=False,
            )
            down_samples, mid_sample = film_gate(ratio_emb, down_samples, mid_sample)
            down_samples = tuple(sample.to(dtype=latents.dtype) for sample in down_samples)
            mid_sample = mid_sample.to(dtype=latents.dtype)
            noise_pred = unet(
                latents,
                timestep,
                encoder_hidden_states=prompt_embeds,
                class_labels=ratio_emb,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
            ).sample
            latents = scheduler.step(noise_pred, timestep, latents).prev_sample
            latents = latents.to(dtype=prompt_embeds.dtype)

    image = _vae_decode(vae, latents / vae.config.scaling_factor)[0]

    samples_dir = Path(output_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    image_path = samples_dir / f"image_step_{step:06d}.png"
    image_array = _save_uint8_rgb(image, image_path)

    layout_ids = layouts.argmax(dim=1)[0]
    layout_color = _colorize_labels(layout_ids, palette)
    layout_path = samples_dir / f"layout_step_{step:06d}.png"
    Image.fromarray(layout_color, mode="RGB").save(layout_path)

    if writer is not None:
        writer.add_image("samples/image", image_array, step, dataformats="HWC")
        writer.add_image("samples/layout", layout_color, step, dataformats="HWC")


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    log_with = None if args.report_to == "none" else args.report_to
    if not torch.cuda.is_available() and args.mixed_precision != "no":
        logger.warning("CUDA not available. Disabling mixed precision for CPU training.")
        args.mixed_precision = "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )
    if accelerator.device.type == "cpu" and args.mixed_precision != "no":
        logger.warning("CPU device detected. Forcing mixed_precision=no to avoid CPU fp16 issues.")
        args.mixed_precision = "no"
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

    train_dataset = _resolve_dataset(args, num_classes)

    def collate_fn(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        layouts = torch.stack([ex["layout_512"] for ex in examples])
        layouts_64 = torch.stack([ex["layout_64"] for ex in examples])
        ratios = torch.stack([ex["ratios"] for ex in examples])
        return {"pixel_values": pixel_values, "layout_512": layouts, "layout_64": layouts_64, "ratios": ratios}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    sample_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    _ensure_identity_class_embedding(unet)
    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=num_classes)
    _ensure_identity_class_embedding(controlnet)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.eval()
    vae.eval()
    text_encoder.eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            if version.parse(torch.__version__) >= version.parse("1.13"):
                unet.enable_xformers_memory_efficient_attention()
                controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    time_embed_dim = infer_time_embed_dim_from_config(unet.config.block_out_channels)
    ratio_projector = RatioProjector(num_classes, time_embed_dim)
    down_channels = []
    mid_channels = None
    if hasattr(controlnet, "controlnet_down_blocks"):
        down_channels = [int(block.out_channels) for block in controlnet.controlnet_down_blocks]
    if hasattr(controlnet, "controlnet_mid_block") and hasattr(controlnet.controlnet_mid_block, "out_channels"):
        mid_channels = int(controlnet.controlnet_mid_block.out_channels)
    if not down_channels or mid_channels is None:
        layers = int(getattr(controlnet.config, "layers_per_block", 1))
        block_out_channels = list(getattr(controlnet.config, "block_out_channels", []))
        down_block_types = list(getattr(controlnet.config, "down_block_types", []))
        if not block_out_channels or not down_block_types:
            raise ValueError("Could not infer ControlNet residual channels for PerChannelResidualFiLMGate.")
        down_channels = [int(block_out_channels[0])]
        for idx, out_ch in enumerate(block_out_channels):
            down_channels.extend([int(out_ch)] * layers)
            if idx != len(block_out_channels) - 1:
                down_channels.append(int(out_ch))
        mid_channels = int(block_out_channels[-1])

    film_gate = PerChannelResidualFiLMGate(
        time_embed_dim,
        down_channels=down_channels,
        mid_channels=mid_channels,
        init_zero=True,
    )

    ema_controlnet = None
    ema_ratio_projector = None
    ema_film_gate = None
    if args.use_ema:
        ema_controlnet = EMAModel(
            controlnet.parameters(),
            decay=args.ema_decay,
            model_cls=ControlNetModel,
            model_config=controlnet.config,
        )
        ema_ratio_projector = EMAModel(ratio_projector.parameters(), decay=args.ema_decay)
        ema_film_gate = EMAModel(film_gate.parameters(), decay=args.ema_decay)

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size

    optimizer = torch.optim.AdamW(
        chain(controlnet.parameters(), ratio_projector.parameters(), film_gate.parameters()),
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

    controlnet, ratio_projector, film_gate, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, ratio_projector, film_gate, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.device.type == "cuda":
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_controlnet.to(accelerator.device)
        ema_ratio_projector.to(accelerator.device)
        ema_film_gate.to(accelerator.device)

    text_inputs = tokenizer(
        [args.prompt],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)

    uncond_inputs = tokenizer(
        [""],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(accelerator.device))[0]
    uncond_embeds = uncond_embeds.to(dtype=weight_dtype)

    palette = build_palette(class_names, num_classes, dataset=args.dataset)

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        if os.path.basename(args.resume_from_checkpoint).startswith("checkpoint-"):
            global_step = int(os.path.basename(args.resume_from_checkpoint).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()
        ratio_projector.train()
        film_gate.train()
        for batch in train_dataloader:
            with accelerator.accumulate(controlnet):
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                layouts_true = batch["layout_512"].to(device=accelerator.device, dtype=weight_dtype)
                layouts_64 = batch["layout_64"].to(device=accelerator.device, dtype=weight_dtype)
                layouts_coarse = F.interpolate(layouts_64, size=(args.image_size, args.image_size), mode="nearest")
                bsz = layouts_true.shape[0]
                use_coarse = (torch.rand((bsz,), device=accelerator.device) < args.mask_mix_prob).view(bsz, 1, 1, 1)
                layouts = torch.where(use_coarse, layouts_coarse, layouts_true)
                ratios = batch["ratios"].to(device=accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                if args.noise_offset and args.noise_offset > 0:
                    noise = noise + args.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1),
                        device=noise.device,
                        dtype=noise.dtype,
                    )
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                ratio_emb = ratio_projector(ratios)
                prompt_batch = prompt_embeds.repeat(bsz, 1, 1)
                if args.cfg_dropout_prob and args.cfg_dropout_prob > 0:
                    drop_txt = torch.rand((bsz,), device=accelerator.device) < args.cfg_dropout_prob
                    if drop_txt.any():
                        prompt_batch[drop_txt] = uncond_embeds.expand(int(drop_txt.sum().item()), -1, -1)

                if args.cond_dropout_prob and args.cond_dropout_prob > 0:
                    drop_cond = torch.rand((bsz,), device=accelerator.device) < args.cond_dropout_prob
                    if drop_cond.any():
                        layouts[drop_cond] = 0.0
                        ratio_emb[drop_cond] = 0.0

                down_samples, mid_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_batch,
                    controlnet_cond=layouts,
                    class_labels=ratio_emb,
                    return_dict=False,
                )
                down_samples, mid_sample = film_gate(ratio_emb, down_samples, mid_sample)
                down_samples = tuple(sample.to(dtype=noisy_latents.dtype) for sample in down_samples)
                mid_sample = mid_sample.to(dtype=noisy_latents.dtype)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_batch,
                    class_labels=ratio_emb,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                ).sample

                if args.snr_gamma and args.snr_gamma > 0:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                else:
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        chain(controlnet.parameters(), ratio_projector.parameters(), film_gate.parameters()),
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_controlnet.step(controlnet.parameters())
                    ema_ratio_projector.step(ratio_projector.parameters())
                    ema_film_gate.step(film_gate.parameters())
                progress_bar.update(1)
                global_step += 1
                if global_step % args.logging_steps == 0:
                    accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        layouts_sample = layouts[:1].detach()
                        ratios_sample = ratios[0].detach()
                        was_training = controlnet.training
                        controlnet.eval()
                        ratio_projector.eval()
                        film_gate.eval()
                        _log_controlnet_sample(
                            accelerator=accelerator,
                            controlnet=accelerator.unwrap_model(controlnet),
                            unet=unet,
                            vae=vae,
                            prompt_embeds=prompt_embeds,
                            ratio_projector=accelerator.unwrap_model(ratio_projector),
                            film_gate=accelerator.unwrap_model(film_gate),
                            scheduler=sample_scheduler,
                            layouts=layouts_sample,
                            ratios=ratios_sample,
                            output_dir=args.output_dir,
                            step=global_step,
                            seed=args.sample_seed,
                            palette=palette,
                            num_inference_steps=args.sample_num_inference_steps,
                        )
                        if was_training:
                            controlnet.train()
                            ratio_projector.train()
                            film_gate.train()

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if args.use_ema:
        ema_controlnet.copy_to(controlnet.parameters())
        ema_ratio_projector.copy_to(ratio_projector.parameters())
        ema_film_gate.copy_to(film_gate.parameters())
    if accelerator.is_main_process:
        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
        unwrapped_ratio = accelerator.unwrap_model(ratio_projector)
        unwrapped_gate = accelerator.unwrap_model(film_gate)
        unwrapped_controlnet.save_pretrained(os.path.join(args.output_dir, "controlnet"))
        torch.save(unwrapped_ratio.state_dict(), os.path.join(args.output_dir, "ratio_projector.bin"))
        torch.save(unwrapped_gate.state_dict(), os.path.join(args.output_dir, "film_gate.bin"))
        with open(os.path.join(args.output_dir, "class_names.json"), "w", encoding="utf-8") as handle:
            json.dump(class_names, handle, indent=2)
        with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset": args.dataset,
                    "num_classes": num_classes,
                    "image_size": args.image_size,
                    "layout_size": args.layout_size,
                    "ignore_index": args.ignore_index,
                    "prompt": args.prompt,
                    "base_model": args.pretrained_model_name_or_path,
                    "mask_mix_prob": args.mask_mix_prob,
                    "cfg_dropout_prob": args.cfg_dropout_prob,
                    "cond_dropout_prob": args.cond_dropout_prob,
                    "snr_gamma": args.snr_gamma,
                    "noise_offset": args.noise_offset,
                    "use_ema": args.use_ema,
                    "ema_decay": args.ema_decay,
                },
                handle,
                indent=2,
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
