#!/usr/bin/env python
# coding=utf-8

"""Train a layout DDPM conditioned on class ratios."""

import argparse
import json
import logging
import math
import os
from itertools import chain
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

from dataset_loveda import GenericSegDataset, LoveDADataset, load_class_names
from ratio_conditioning import RatioProjector, infer_time_embed_dim_from_config


check_min_version("0.37.0.dev0")

logger = get_logger(__name__, log_level="INFO")

if is_wandb_available():
    import wandb  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Train a layout DDPM conditioned on class ratios.")
    parser.add_argument("--data_root", type=str, required=True, help="Root folder for the dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="generic",
        choices=["loveda", "generic"],
        help="Dataset type to use.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/layout_ddpm")
    parser.add_argument("--image_size", type=int, default=512, help="Dataset image size (square).")
    parser.add_argument("--layout_size", type=int, default=64, help="Layout diffusion size (square).")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes in the dataset.")
    parser.add_argument("--class_names_json", type=str, default=None)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--loveda_split", type=str, default="Train")
    parser.add_argument("--loveda_domains", type=str, default="Urban,Rural")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=20000)
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
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
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
        )
    return GenericSegDataset(
        args.data_root,
        image_size=args.image_size,
        num_classes=num_classes,
        ignore_index=None,
        return_layouts=True,
    )


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

    train_dataset = _resolve_dataset(args, num_classes)

    def collate_fn(examples):
        layouts = torch.stack([ex["layout_64"] for ex in examples])
        ratios = torch.stack([ex["ratios"] for ex in examples])
        return {"layout_64": layouts, "ratios": ratios}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    unet = UNet2DModel(
        sample_size=args.layout_size,
        in_channels=num_classes,
        out_channels=num_classes,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        class_embed_type="identity",
    )

    time_embed_dim = infer_time_embed_dim_from_config(unet.config.block_out_channels)
    ratio_projector = RatioProjector(num_classes, time_embed_dim)

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)

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

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        if Path(args.resume_from_checkpoint).name.startswith("checkpoint-"):
            global_step = int(Path(args.resume_from_checkpoint).name.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        ratio_projector.train()
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                layouts = batch["layout_64"]
                ratios = batch["ratios"]
                layouts = layouts.to(dtype=torch.float32)
                ratios = ratios.to(dtype=torch.float32)
                layouts = layouts * 2.0 - 1.0

                noise = torch.randn_like(layouts)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (layouts.shape[0],), device=layouts.device
                ).long()
                noisy_layouts = noise_scheduler.add_noise(layouts, noise, timesteps)

                ratio_emb = ratio_projector(ratios)
                noise_pred = unet(noisy_layouts, timesteps, class_labels=ratio_emb).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

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
                    accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

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
        noise_scheduler.save_pretrained(os.path.join(args.output_dir, "scheduler"))
        with open(os.path.join(args.output_dir, "class_names.json"), "w", encoding="utf-8") as handle:
            json.dump(class_names, handle, indent=2)
        with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset": args.dataset,
                    "num_classes": num_classes,
                    "layout_size": args.layout_size,
                    "image_size": args.image_size,
                    "ignore_index": args.ignore_index,
                    "num_train_timesteps": args.num_train_timesteps,
                },
                handle,
                indent=2,
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
