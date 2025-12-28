# SyntheticGen: Ratio-Conditioned Layout + ControlNet Image Generation

This repo implements a two-stage pipeline:

- Stage A: a ratio-conditioned layout DDPM that generates a multi-class layout (K channels) at 64x64 (or 128/256).
- Stage B: Stable Diffusion + ControlNet conditioned on the layout, with FiLM-style ratio gating.

The layout generator enforces class ratios during training (explicit ratio loss) and can add an optional boundary/edge loss for sharper maps.
Sampling supports optional histogram guidance, CFG, and a ControlNet conditioning-scale schedule.

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

Stage A (layout DDPM):

```bash
python src/scripts/train_layout_ddpm.py \
  --data_root /path/to/loveda \
  --dataset loveda \
  --output_dir outputsimproved/layout_ddpm \
  --layout_size 128 \
  --checkpointing_steps 500 \
  --sample_num_inference_steps 50 \
  --lambda_ratio 1.0 \
  --lambda_edge 0.25 \
  --ratio_temp 1.0
```

Optional (recommended) Stage A.5: domain-adapt SD UNet (remote-sensing style)

```bash
python src/scripts/train_sd_unet_adapter.py \
  --pretrained_model_name_or_path /path/to/sd-v1-5 \
  --data_root /path/to/loveda \
  --dataset loveda \
  --output_dir outputsimproved/sd_unet_adapter \
  --unet_trainable_up_blocks 2
```

Stage B (ControlNet + FiLM ratio conditioning):

```bash
python src/scripts/train_controlnet_ratio.py \
  --pretrained_model_name_or_path /path/to/sd-v1-5 \
  --unet_init outputsimproved/sd_unet_adapter/unet \
  --data_root /path/to/loveda \
  --dataset loveda \
  --output_dir outputsimproved/controlnet_ratio \
  --unet_trainable_up_blocks 2 \
  --unet_unfreeze_step 5000 \
  --unet_lr 2e-6
```

You can also use the wrapper:

```bash
python src/scripts/main.py train_layout -- --data_root /path/to/loveda
python src/scripts/main.py train_sd_adapter -- --data_root /path/to/loveda --base_model /path/to/sd-v1-5
python src/scripts/main.py train_controlnet -- --data_root /path/to/loveda --base_model /path/to/sd-v1-5
```

## Sampling

From scratch (ratios -> layout -> image):

```bash
python src/scripts/sample_pair.py \
  --layout_ckpt outputsimproved/layout_ddpm \
  --controlnet_ckpt outputsimproved/controlnet_ratio \
  --base_model /path/to/sd-v1-5 \
  --guidance_scale 5.0 \
  --controlnet_scale_start 1.0 \
  --controlnet_scale_end 0.5 \
  --ratios "0.05,0.2,0.1,0.05,0.1,0.25,0.25" \
  --save_dir outputsimproved/sample_pair
```

Img2img with a provided mask:

```bash
python src/scripts/sample_pair.py \
  --layout_ckpt outputs/layout_ddpm \
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
  --layout_ckpt outputs/layout_ddpm \
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
- If `outputs/controlnet_ratio/unet` exists (UNet adapter was trained during Stage B), sampling uses it automatically.

## Evaluation

Ratio control metrics:

```bash
python src/scripts/eval_ratio_control.py \
  --layout_ckpt outputs/layout_ddpm \
  --ratios "0.05,0.2,0.1,0.05,0.1,0.25,0.25" \
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
