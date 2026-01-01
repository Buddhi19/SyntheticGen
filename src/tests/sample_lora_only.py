#!/usr/bin/env python3

import argparse
import os

import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import is_peft_available
from transformers import CLIPTextModel, CLIPTokenizer


def parse_args():
    ap = argparse.ArgumentParser(description="Quick LoRA-only sampling sanity check (SD1.5 UNet LoRA).")
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--lora_dir", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="a satellite image, aerial orthophoto, remote sensing, nadir view")
    ap.add_argument("--outdir", type=str, default="lora_samples")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--lora_scale", type=float, default=1.0)
    ap.add_argument("--weight_name", type=str, default="pytorch_lora_weights.safetensors")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    return ap.parse_args()


def _build_pipeline(base_model: str) -> StableDiffusionPipeline:
    if os.path.isdir(base_model) and not os.path.isfile(os.path.join(base_model, "model_index.json")):
        tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=torch.float16)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(base_model, subfolder="scheduler")
        return StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    return StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    if not is_peft_available():
        raise ImportError("PEFT is required for LoRA loading. Install it with: pip install peft")

    pipe = _build_pipeline(args.base_model).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.unet.load_attn_procs(args.lora_dir, weight_name=args.weight_name)

    generator = torch.Generator("cuda").manual_seed(int(args.seed))
    cross_kwargs = None
    if args.lora_scale is not None and float(args.lora_scale) != 1.0:
        cross_kwargs = {"scale": float(args.lora_scale)}

    out = pipe(
        prompt=args.prompt,
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.cfg),
        height=int(args.height),
        width=int(args.width),
        num_images_per_prompt=int(args.num),
        generator=generator,
        cross_attention_kwargs=cross_kwargs,
    )

    for idx, image in enumerate(out.images):
        image.save(os.path.join(args.outdir, f"img_{idx:02d}.png"))

    print(f"Saved {len(out.images)} images to: {args.outdir}")


if __name__ == "__main__":
    main()
