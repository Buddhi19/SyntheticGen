#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a Stable Diffusion UNet to jointly denoise image + class mask latents."""

import argparse
import json
import logging
import math
import os

import accelerate
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from dataset_loveda import GenericSegDataset, LoveDADataset, load_class_names


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.37.0.dev0")

logger = get_logger(__name__, log_level="INFO")


if is_wandb_available():
    import wandb  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Train a joint image+mask diffusion model in latent space.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="seg-mask-diffusion",
        help="Where to store checkpoints and final model.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument("--data_root", type=str, required=True, help="Root folder for the dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="generic",
        choices=["loveda", "generic"],
        help="Dataset type to use.",
    )
    parser.add_argument("--image_size", type=int, default=512, help="Training image size (square).")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes in the dataset.")
    parser.add_argument(
        "--class_names_json",
        type=str,
        default=None,
        help="Path to JSON list/dict mapping class ids to names.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="a remote sensing image and mask of {class_name}",
        help="Prompt template used for class conditioning.",
    )
    parser.add_argument("--sample_present_class", action="store_true", default=True)
    parser.add_argument("--no_sample_present_class", action="store_false", dest="sample_present_class")
    parser.add_argument("--ignore_index", type=int, default=255, help="Ignore index in segmentation labels.")
    parser.add_argument("--loveda_split", type=str, default="Train", help="LoveDA split name.")
    parser.add_argument(
        "--loveda_domains",
        type=str,
        default="Urban,Rural",
        help="Comma-separated list of LoveDA domains to include.",
    )
    parser.add_argument(
        "--mask_loss_weight",
        type=float,
        default=1.0,
        help="Weight (lambda) applied to the mask denoising loss.",
    )
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--scale_lr", action="store_true", help="Scale the learning rate by batch size.")
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
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
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


def _select_class_ids(labels, num_classes, ignore_index, sample_present_class):
    class_ids = []
    for label in labels:
        if sample_present_class:
            unique = torch.unique(label)
            if ignore_index is not None:
                unique = unique[unique != ignore_index]
            if unique.numel() > 0:
                idx = torch.randint(0, unique.numel(), (1,), device=label.device)
                class_id = unique[idx].item()
            else:
                class_id = torch.randint(0, num_classes, (1,), device=label.device).item()
        else:
            class_id = torch.randint(0, num_classes, (1,), device=label.device).item()
        class_ids.append(class_id)
    return torch.tensor(class_ids, device=labels.device, dtype=torch.long)


def main():
    args = parse_args()

    if args.image_size % 8 != 0:
        raise ValueError("--image_size must be divisible by 8 for Stable Diffusion latents.")

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    log_with = None if args.report_to == "none" else args.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )

    if log_with == "wandb" and not is_wandb_available():
        raise ImportError("wandb is not installed. Please install it or pass --report_to tensorboard.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    in_channels = 5
    out_channels = 5
    unet.register_to_config(in_channels=in_channels, out_channels=out_channels)
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(
            in_channels, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        if unet.conv_in.bias is not None:
            new_conv_in.bias.copy_(unet.conv_in.bias)
        unet.conv_in = new_conv_in

        new_conv_out = torch.nn.Conv2d(
            unet.conv_out.in_channels,
            out_channels,
            unet.conv_out.kernel_size,
            unet.conv_out.stride,
            unet.conv_out.padding,
        )
        new_conv_out.weight.zero_()
        new_conv_out.weight[:4, :, :, :].copy_(unet.conv_out.weight)
        if unet.conv_out.bias is not None:
            new_conv_out.bias.zero_()
            new_conv_out.bias[:4].copy_(unet.conv_out.bias)
        unet.conv_out = new_conv_out

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "If you observe problems, update to at least 0.0.17."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.") from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    class_names, num_classes = load_class_names(args.class_names_json, args.num_classes, args.dataset)
    args.num_classes = num_classes

    if args.dataset == "loveda":
        if args.ignore_index != 255:
            logger.warning("LoveDA labels are remapped to 0..6; overriding ignore_index to 255.")
            args.ignore_index = 255
        domains = [domain.strip() for domain in args.loveda_domains.split(",") if domain.strip()]
        train_dataset = LoveDADataset(
            args.data_root,
            image_size=args.image_size,
            split=args.loveda_split,
            domains=domains,
            ignore_index=args.ignore_index,
        )
    else:
        train_dataset = GenericSegDataset(args.data_root, image_size=args.image_size)

    def collate_fn(examples):
        pixel_values = torch.stack([example["image"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        labels = torch.stack([example["label"] for example in examples]).long()
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps = args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                for model in models:
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    text_encoder.eval()

    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "class_names.json"), "w", encoding="utf-8") as handle:
            json.dump(class_names, handle, indent=2)
        with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "prompt_template": args.prompt_template,
                    "num_classes": args.num_classes,
                    "ignore_index": args.ignore_index,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "mask_loss_weight": args.mask_loss_weight,
                },
                handle,
                indent=2,
            )

    if accelerator.is_main_process and log_with is not None:
        accelerator.init_trackers("seg-mask-diffusion", config=vars(args))

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("Running training")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        log_steps = 0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                labels = batch["labels"].to(accelerator.device)

                with torch.no_grad():
                    image_latents = vae.encode(pixel_values).latent_dist.sample()
                    image_latents = image_latents * vae.config.scaling_factor

                class_ids = _select_class_ids(
                    labels, num_classes, args.ignore_index, args.sample_present_class
                )
                class_names_batch = [class_names[class_id] for class_id in class_ids.tolist()]
                prompts = [
                    args.prompt_template.format(class_name=class_name, class_id=class_id)
                    for class_name, class_id in zip(class_names_batch, class_ids.tolist())
                ]
                text_inputs = tokenizer(
                    prompts,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

                mask = (labels == class_ids[:, None, None]).float().unsqueeze(1)
                latent_h, latent_w = image_latents.shape[-2:]
                mask_latents = F.interpolate(mask, size=(latent_h, latent_w), mode="nearest")
                mask_latents = mask_latents * 2.0 - 1.0
                mask_latents = mask_latents.to(dtype=weight_dtype)

                noise_image = torch.randn_like(image_latents)
                noise_mask = torch.randn_like(mask_latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (image_latents.shape[0],), device=image_latents.device
                ).long()
                noisy_image_latents = noise_scheduler.add_noise(image_latents, noise_image, timesteps)
                noisy_mask_latents = noise_scheduler.add_noise(mask_latents, noise_mask, timesteps)
                model_input = torch.cat([noisy_image_latents, noisy_mask_latents], dim=1)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target_image = noise_image
                    target_mask = noise_mask
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target_image = noise_scheduler.get_velocity(image_latents, noise_image, timesteps)
                    target_mask = noise_scheduler.get_velocity(mask_latents, noise_mask, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                noise_pred = unet(model_input, timesteps, encoder_hidden_states).sample
                noise_pred_image = noise_pred[:, :4, :, :]
                noise_pred_mask = noise_pred[:, 4:5, :, :]
                loss_image = F.mse_loss(noise_pred_image.float(), target_image.float(), reduction="mean")
                loss_mask = F.mse_loss(noise_pred_mask.float(), target_mask.float(), reduction="mean")
                loss = loss_image + args.mask_loss_weight * loss_mask

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                log_steps += 1
                if global_step % args.logging_steps == 0:
                    avg_log_loss = train_loss / max(1, log_steps)
                    accelerator.log({"train_loss": avg_log_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    train_loss = 0.0
                    log_steps = 0

                if args.checkpointing_steps is not None and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [
                                d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")
                            ]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                for checkpoint in checkpoints[:num_to_remove]:
                                    path = os.path.join(args.output_dir, checkpoint)
                                    logger.info(f"Removing old checkpoint {path}")
                                    for root, dirs, files in os.walk(path, topdown=False):
                                        for name in files:
                                            os.remove(os.path.join(root, name))
                                        for name in dirs:
                                            os.rmdir(os.path.join(root, name))
                                    os.rmdir(path)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        unwrapped_unet = unwrap_model(unet)
        unwrapped_unet.save_pretrained(os.path.join(args.output_dir, "unet"))
        vae.save_pretrained(os.path.join(args.output_dir, "vae"))
        text_encoder.save_pretrained(os.path.join(args.output_dir, "text_encoder"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
        noise_scheduler.save_pretrained(os.path.join(args.output_dir, "scheduler"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
