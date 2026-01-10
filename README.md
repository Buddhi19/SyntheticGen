# SyntheticGen: Ratio-Conditioned Layout + ControlNet Image Generation

This repo implements a two-stage pipeline:

- Stage A: a ratio-conditioned layout D3PM (categorical diffusion) that generates a multi-class label map at `layout_size`.
- Stage B: Stable Diffusion + ControlNet conditioned on the layout, with FiLM-style ratio gating.

The layout generator enforces class ratios during training (explicit ratio loss), and sampling supports optional histogram guidance.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset formats

### LoveDA
Expected structure (either Urban/Rural or a subset):

```
LoveDA/
  Train/
    Train/            # some LoveDA releases have an extra nesting level
      Urban/
        images_png/
        masks_png/
      Rural/
        images_png/
        masks_png/
    Urban/            # also supported (no extra nesting)
      images_png/
      masks_png/
    Rural/
      images_png/
      masks_png/
  Val/
    ...
```

The default class list for LoveDA is used unless you pass `--class_names_json`.

### Generic folder dataset
Put matching image/mask files under `images/` and `masks/` (filenames must match by stem):

```
my_data/
  images/
    0001.png
  masks/
    0001.png
```

Masks should be single-channel label maps (integer class ids) or paletted PNGs.

## Training

## YAML configs

All scripts accept `--config` pointing to a YAML/JSON file (keys match CLI arg names). Example configs live in `configs/`.

```bash
# Stage A (D3PM) on 2 GPUs (6,7)
CUDA_VISIBLE_DEVICES=6,7 python3 -m accelerate.commands.launch --multi_gpu --num_processes 2 --main_process_port 29507 \
  src/scripts/train_layout_d3pm.py --config configs/train_layout_d3pm_masked_sparse_80k.yaml

# Ratio prior
python src/scripts/compute_ratio_prior.py --config configs/compute_ratio_prior_loveda_train.yaml

# Stage B (ControlNet)
CUDA_VISIBLE_DEVICES=6,7 python3 -m accelerate.commands.launch --multi_gpu --num_processes 2 --main_process_port 29509 \
  src/scripts/train_controlnet_ratio.py --config configs/train_controlnet_ratio_loveda_1024.yaml
```

Stage A (layout D3PM):

```bash
python src/scripts/train_layout_d3pm.py \
  --data_root /path/to/loveda \
  --dataset loveda \
  --output_dir outputs/layout_d3pm \
  --checkpointing_steps 500 \
  --sample_num_inference_steps 50 \
  --lambda_ratio 1.0 \
  --ratio_temp 1.0
```

Optional: train Stage A with sparse (partial) ratio conditioning:

```bash
python src/scripts/compute_ratio_prior.py \
  --data_root /path/to/loveda \
  --dataset loveda \
  --image_size 1024 \
  --output_path outputsV2/ratio_prior.json

python src/scripts/train_layout_d3pm.py \
  --data_root /path/to/loveda \
  --dataset loveda \
  --output_dir outputs/layout_d3pm_masked \
  --image_size 1024 \
  --layout_size 256 \
  --ratio_conditioning masked \
  --p_keep 0.3 \
  --known_count_max 3 \
  --lambda_ratio 1.0 \
  --lambda_prior 0.1 \
  --ratio_prior_json outputsV2/ratio_prior.json
```

Stage B (ControlNet + FiLM ratio conditioning):

```bash
python src/scripts/train_controlnet_ratio.py \
  --pretrained_model_name_or_path /path/to/sd-v1-5 \
  --data_root /path/to/loveda \
  --dataset loveda \
  --output_dir outputs/controlnet_ratio
```

You can also use the wrapper:

```bash
python src/scripts/main.py train_layout -- --data_root /path/to/loveda
python src/scripts/main.py train_controlnet -- --data_root /path/to/loveda --base_model /path/to/sd-v1-5
```

## Sampling

From scratch (ratios -> layout -> image):

```bash
python src/scripts/sample_pair.py \
  --layout_ckpt outputs/layout_d3pm \
  --controlnet_ckpt outputs/controlnet_ratio \
  --base_model /path/to/sd-v1-5 \
  --ratios "0.05,0.2,0.1,0.05,0.1,0.25,0.25" \
  --save_dir outputs/sample_pair
```

Sparse ratio constraints (works with Stage A checkpoints trained with `--ratio_conditioning masked`):

```bash
python src/scripts/sample_pair.py \
  --layout_ckpt outputs/layout_d3pm_masked \
  --controlnet_ckpt outputs/controlnet_ratio \
  --base_model /path/to/sd-v1-5 \
  --ratios "water:0.15,agriculture:0.10" \
  --save_dir outputs/sample_pair_sparse
```

Example (specific checkpoint + single-class ratio):

```bash
CUDA_VISIBLE_DEVICES=7 python src/scripts/sample_pair.py --config configs/sample_pair_ckpt40000_building0.4.yaml

CUDA_VISIBLE_DEVICES=7 python /data/inr/llm/DIFF_CD/Diffusor/SyntheticGen/src/scripts/sample_pair.py \
  --layout_ckpt /data/inr/llm/DIFF_CD/Diffusor/outputsV3/layout_d3pm_masked_sparse_80k \
  --controlnet_ckpt /data/inr/llm/DIFF_CD/Diffusor/outputsV3/controlnet_ratio_lora_ckpt18000_layout80000/checkpoint-40000 \
  --base_model /home/nvidia/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14 \
  --lora_path /data/inr/llm/DIFF_CD/Diffusor/outputsV2/lora_loveda_sd15_r8/checkpoint-29000 \
  --lora_weight_name pytorch_lora_weights.safetensors \
  --lora_scale 1.0 \
  --save_dir /data/inr/llm/DIFF_CD/Diffusor/outputsV3/results_generator/gpu7 \
  --ratios "building:0.4" \
  --prompt "a high-resolution satellite image" \
  --image_size 1024 \
  --num_inference_steps_layout 50 \
  --num_inference_steps_image 30 \
  --guidance_scale 1.0 \
  --guidance_rescale 0.0 \
  --control_scale 1.0 \
  --seed 40000 \
  --dtype fp16 \
  --device cuda:0 \
  --sampler ddim
```

Img2img with a provided mask:

```bash
python src/scripts/sample_pair.py \
  --layout_ckpt outputs/layout_d3pm \
  --controlnet_ckpt outputs/controlnet_ratio \
  --base_model /path/to/sd-v1-5 \
  --init_image /path/to/image.png \
  --init_mask /path/to/mask.png \
  --mask_format loveda_raw \
  --ratios "0.05,0.2,0.1,0.05,0.1,0.25,0.25" \
  --save_dir outputs/edited_pair
```

Image-only editing (auto mask init):

```bash
python src/scripts/sample_pair.py \
  --layout_ckpt outputs/layout_d3pm \
  --controlnet_ckpt outputs/controlnet_ratio \
  --base_model /path/to/sd-v1-5 \
  --init_image /path/to/image.png \
  --seg_ckpt /path/to/segmentation_state.pt \
  --seg_arch simple \
  --ratios "0.05,0.2,0.1,0.05,0.1,0.25,0.25" \
  --save_dir outputs/edited_pair
```

Notes:
- `--mask_format` supports `indexed` (0..K-1 with ignore) and `loveda_raw` (0=ignore, 1..K).
- `--ignore_index` defaults to 255 and is respected when building one-hot layouts.

## Evaluation

Ratio control metrics:

```bash
python src/scripts/eval_ratio_control.py \
  --layout_ckpt outputs/layout_d3pm \
  --data_root /path/to/loveda \
  --dataset loveda \
  --num_samples 100
```

Downstream segmentation utility:

```bash
python src/scripts/eval_downstream_segmentation.py \
  --real_data_root /path/to/loveda \
  --synthetic_data_root /path/to/synth_dataset \
  --dataset loveda \
  --output_path outputs/eval_downstream_segmentation.json
```

## Outputs

Training scripts write `training_config.json` and `class_names.json` into the checkpoint directories.
Sampling writes `image.png`, `layout.png`, and `metadata.json` to the output directory.
