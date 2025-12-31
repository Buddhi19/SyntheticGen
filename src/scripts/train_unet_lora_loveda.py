#!/usr/bin/env python
# coding=utf-8

"""
Train UNet LoRA on LoveDA images (no ControlNet).
Saves LoRA weights using unet.save_attn_procs(output_dir).
Load later with unet.load_attn_procs(lora_dir, weight_name="pytorch_lora_weights.safetensors").
"""

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_peft_available
from tqdm.auto import tqdm

try:
    from .dataset_loveda import LoveDADataset, GenericSegDataset
except ImportError:  # direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.scripts.dataset_loveda import LoveDADataset, GenericSegDataset


check_min_version("0.36.0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet LoRA on LoveDA images.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True, help="LoveDA root (or generic root).")
    parser.add_argument("--dataset", type=str, default="loveda", choices=["loveda", "generic"])
    parser.add_argument("--split", type=str, default="Train")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--resolution", type=int, default=512, help="Images are resized to this size.")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_scale", type=float, default=1.0, help="Optional LoRA scale for sampling usage.")
    parser.add_argument("--snr_gamma", type=float, default=0.0, help="Set >0 to enable min-SNR loss reweighting.")

    parser.add_argument("--prompt", type=str, default="a satellite image")
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.1, help="Drop text conditioning sometimes.")

    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=50)

    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--num_classes", type=int, default=7, help="Generic dataset class count (unused when loveda).")
    return parser.parse_args()


def add_lora_adapter(unet: UNet2DConditionModel, rank: int):
    if not is_peft_available():
        raise ImportError("PEFT is not available. Install it with: pip install peft")
    from peft import LoraConfig

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
    )
    unet.add_adapter(lora_config)
    lora_params = [param for param in unet.parameters() if param.requires_grad]
    if not lora_params:
        raise RuntimeError("No trainable LoRA parameters found after adapter injection.")
    return lora_params


def collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples]).contiguous()
    return {"pixel_values": pixel_values}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with=None,
        project_dir=args.output_dir,
    )

    logging.basicConfig(level=logging.INFO)
    logger.info(accelerator.state)

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    lora_params = add_lora_adapter(unet, rank=args.lora_rank)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.dataset == "loveda":
        dataset = LoveDADataset(
            root=args.data_root,
            split=args.split,
            image_size=args.resolution,
            return_layouts=False,
        )
    else:
        dataset = GenericSegDataset(
            root=args.data_root,
            image_size=args.resolution,
            num_classes=args.num_classes,
            ignore_index=255,
            return_layouts=False,
        )

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        max_train_steps = args.max_train_steps
        args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(dtype=weight_dtype)
    if weight_dtype != torch.float32:
        for param in lora_params:
            param.data = param.data.float()

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

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    for epoch in range(args.num_train_epochs):
        unet.train()
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompt_batch = prompt_embeds.repeat(bsz, 1, 1)
                if args.cfg_dropout_prob and args.cfg_dropout_prob > 0:
                    drop_txt = torch.rand((bsz,), device=accelerator.device) < args.cfg_dropout_prob
                    if drop_txt.any():
                        prompt_batch[drop_txt] = uncond_embeds.expand(int(drop_txt.sum().item()), -1, -1)

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_batch,
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction_type={noise_scheduler.config.prediction_type}")

                if args.snr_gamma and args.snr_gamma > 0:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack(
                        [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                    ).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    else:
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if accelerator.is_local_main_process:
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                if accelerator.is_main_process and (global_step % args.logging_steps == 0):
                    logger.info("epoch=%d step=%d loss=%.4f", epoch, global_step, loss.item())

                if (
                    accelerator.is_main_process
                    and args.checkpointing_steps
                    and (global_step % args.checkpointing_steps == 0)
                ):
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    unwrapped_unet = accelerator.unwrap_model(unet).to(torch.float32)
                    unwrapped_unet.save_attn_procs(ckpt_dir)
                    logger.info("Saved LoRA checkpoint: %s", ckpt_dir)

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet).to(torch.float32)
        unwrapped_unet.save_attn_procs(args.output_dir)
        logger.info("Saved final LoRA to: %s", args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
