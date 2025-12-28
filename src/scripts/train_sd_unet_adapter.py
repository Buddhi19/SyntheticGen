#!/usr/bin/env python
# coding=utf-8

"""Domain-adapt the Stable Diffusion UNet on remote-sensing images (UNet adapter fine-tuning)."""

import argparse
import json
import logging
import math
import os
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

try:
    from .dataset_loveda import GenericSegDataset, LoveDADataset
except ImportError:  # direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.scripts.dataset_loveda import GenericSegDataset, LoveDADataset


check_min_version("0.36.0")
logger = get_logger(__name__, log_level="INFO")

if is_wandb_available():
    import wandb  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight SD UNet adapter (domain adaptation).")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    default_output_dir = str(Path(__file__).resolve().parents[2] / "outputsimproved" / "sd_unet_adapter")
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="generic", choices=["loveda", "generic"])
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--loveda_split", type=str, default="Train")
    parser.add_argument("--loveda_domains", type=str, default="Urban,Rural")
    parser.add_argument("--prompt", type=str, default="a high-resolution satellite image")

    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
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
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--unet_trainable_up_blocks", type=int, default=2, help="How many final UNet up-blocks to train.")

    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--sample_num_inference_steps", type=int, default=30)
    parser.add_argument("--sample_seed", type=int, default=0)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def _resolve_dataset(args):
    if args.dataset == "loveda":
        domains = [domain.strip() for domain in args.loveda_domains.split(",") if domain.strip()]
        return LoveDADataset(
            args.data_root,
            image_size=args.image_size,
            split=args.loveda_split,
            domains=domains,
            ignore_index=args.ignore_index,
            num_classes=1,
            return_layouts=False,
        )
    return GenericSegDataset(
        args.data_root,
        image_size=args.image_size,
        num_classes=1,
        ignore_index=None,
        return_layouts=False,
    )


def _select_trainable_unet_params(unet: UNet2DConditionModel, trainable_up_blocks: int) -> Tuple[List[str], List[torch.nn.Parameter]]:
    if trainable_up_blocks < 0:
        raise ValueError("--unet_trainable_up_blocks must be >= 0")

    for param in unet.parameters():
        param.requires_grad = False

    modules: List[torch.nn.Module] = []
    if trainable_up_blocks > 0:
        modules.extend(list(unet.up_blocks[-trainable_up_blocks:]))
    modules.extend([unet.conv_norm_out, unet.conv_out])

    trainable: List[torch.nn.Parameter] = []
    names: List[str] = []
    for module in modules:
        for name, param in module.named_parameters(recurse=True):
            param.requires_grad = True
            trainable.append(param)
            names.append(name)

    return names, trainable


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
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(array, mode="RGB").save(save_path)


def _log_sample(
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    prompt_embeds: torch.Tensor,
    scheduler: DDIMScheduler,
    output_dir: str,
    step: int,
    seed: int,
    num_inference_steps: int,
) -> None:
    try:
        tracker = accelerator.get_tracker("tensorboard")
        writer = getattr(tracker, "writer", None) if tracker is not None else None
    except Exception:
        writer = None

    device = prompt_embeds.device
    sample_dtype = next(unet.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, unet.config.in_channels, 64, 64),
        generator=generator,
        device=device,
        dtype=sample_dtype,
    )
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps, device=device)

    with torch.no_grad(), accelerator.autocast():
        for t in scheduler.timesteps:
            noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample.to(dtype=sample_dtype)

    image = _vae_decode(vae, latents / vae.config.scaling_factor)[0]
    samples_dir = Path(output_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_path = samples_dir / f"image_step_{step:06d}.png"
    _save_uint8_rgb(image, sample_path)

    if writer is not None:
        writer.add_image("samples/image", (image / 2 + 0.5).clamp(0, 1), step, dataformats="CHW")


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

    dataset = _resolve_dataset(args)

    def collate_fn(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        return {"pixel_values": pixel_values}

    dataloader = torch.utils.data.DataLoader(
        dataset,
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

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xformers is not available. Install it or disable --enable_xformers_memory_efficient_attention.")
        unet.enable_xformers_memory_efficient_attention()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    trainable_names, trainable_params = _select_trainable_unet_params(unet, args.unet_trainable_up_blocks)
    if accelerator.is_main_process:
        logger.info(f"Training UNet params: {len(trainable_params)} tensors (up_blocks={args.unet_trainable_up_blocks}).")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, dataloader, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.device.type == "cuda":
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    text_inputs = tokenizer(
        [args.prompt],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids.to(accelerator.device))[0].to(dtype=weight_dtype)

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
        unet.train()
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompt_batch = prompt_embeds.repeat(latents.shape[0], 1, 1)
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=prompt_batch).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.logging_steps == 0:
                    accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        _log_sample(
                            accelerator=accelerator,
                            unet=accelerator.unwrap_model(unet),
                            vae=vae,
                            prompt_embeds=prompt_embeds,
                            scheduler=sample_scheduler,
                            output_dir=args.output_dir,
                            step=global_step,
                            seed=args.sample_seed,
                            num_inference_steps=args.sample_num_inference_steps,
                        )

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(os.path.join(args.output_dir, "unet"))
        with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "base_model": args.pretrained_model_name_or_path,
                    "prompt": args.prompt,
                    "image_size": args.image_size,
                    "unet_trainable_up_blocks": args.unet_trainable_up_blocks,
                    "learning_rate": args.learning_rate,
                },
                handle,
                indent=2,
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
