# SyntheticGen: Joint Image + Mask Diffusion

This example trains a Stable Diffusion UNet to jointly denoise an image latent (4ch) and a single-class binary mask latent (1ch), guided by a text prompt describing the selected class. The UNet operates on a 5-channel latent formed by concatenating the noised image latent and the noised mask latent.

## Setup

Install diffusers from source and the example requirements:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .

cd examples/SyntheticGen
pip install -r requirements.txt
```

Configure Accelerate:

```bash
accelerate config
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

Example (LoveDA):

```bash
accelerate launch train_seg_mask_diffusion.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --data_root /path/to/loveda \
  --dataset loveda \
  --output_dir outputs/sdseg
```

Example (generic dataset):

```bash
accelerate launch train_seg_mask_diffusion.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --data_root /path/to/my_data \
  --dataset generic \
  --num_classes 10 \
  --output_dir outputs/sdseg
```

Key options:
- `--prompt_template "a remote sensing image and mask of {class_name}"`
- `--sample_present_class / --no_sample_present_class`
- `--ignore_index 255`
- `--mask_loss_weight 1.0` (lambda for the mask loss term)

The script writes `class_names.json` and `training_config.json` into `output_dir` for generation reuse.

## Generation

```bash
python generate_joint_image_mask_diffusion.py \
  --checkpoint outputs/sdseg \
  --save_dir outputs/synth \
  --num_samples 100
```

Optional flags:
- `--dataset loveda` (to infer default class names)
- `--class_id <id>` (generate samples for a single class)
- `--scheduler ddim` (default) or `ddpm`
- `--image_size` should match the training resolution
