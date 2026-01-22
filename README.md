<div align="center">

# 🎨 SyntheticGen

### MITIGATING LONG-TAIL BIAS VIN LOVEDA IA PROMPT-CONTROLLED DIFFUSION AUGMENTATION

*Addressing class imbalance in remote sensing datasets through controlled synthetic generation*

[![Paper (yet to come)](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/your-paper-id)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-blue)](https://drive.google.com/drive/folders/14cMpLTgvcLdXhRY0kGhFKpDRMvpok90h?usp=sharing)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 🌟 Overview

**SyntheticGen** tackles the long-tail distribution problem in the LoveDA remote sensing dataset by generating synthetic imagery with *explicit control* over class ratios. Unlike traditional augmentation methods, our two-stage pipeline lets you specify exactly what proportion of each land cover class should appear in your generated images.

<div align="center">
  <img src="docs/results.png" alt="SyntheticGen Results" width="100%">
</div>


---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Buddhi19/SyntheticGen.git
cd syntheticgen

# Install dependencies
pip install -r requirements.txt
```

### Generate Your First Synthetic Image

```bash
# Use a pre-configured example
python src/scripts/sample_pair.py \
  --config configs/sample_pair_ckpt40000_building0.4.yaml
```
---

## 📚 Usage

### Training Pipeline

#### Step 1: Train Layout Generator (Stage A)

Train the D3PM model to generate semantic layouts conditioned on class ratios:

```bash
python src/scripts/train_layout_d3pm.py \
  --config configs/train_layout_d3pm_masked_sparse_80k.yaml
```

#### Step 2: Compute Ratio Prior (Optional)

For sparse ratio conditioning, compute statistics from your training set:

```bash
python src/scripts/compute_ratio_prior.py \
  --config configs/compute_ratio_prior_loveda_train.yaml
```

#### Step 3: Train Image Generator (Stage B)

Train the ControlNet model to synthesize images from layouts:

```bash
python src/scripts/train_controlnet_ratio.py \
  --config configs/train_controlnet_ratio_loveda_1024.yaml
```

### Inference

#### Generate Image-Layout Pairs

```bash
# Using a config file
python src/scripts/sample_pair.py \
  --config configs/sample_pair_ckpt40000_building0.4.yaml

# Override config parameters via CLI
python src/scripts/sample_pair.py \
  --config configs/sample_pair_ckpt40000_building0.4.yaml \
  --ratios "building:0.4,forest:0.3" \
  --save_dir outputs/custom_generation
```

---

## 📁 Data Format

### LoveDA Dataset Structure

```
LoveDA/
├── Train/
│   ├── Urban/
│   │   ├── images_png/
│   │   └── masks_png/
│   └── Rural/
│       ├── images_png/
│       └── masks_png/
└── Val/
    ├── Urban/
    └── Rural/
```

### Generic Dataset Structure

For custom datasets, organize as:

```
your_dataset/
├── images/
│   ├── image_001.png
│   └── image_002.png
└── masks/
    ├── image_001.png  # Label map with matching stem
    └── image_002.png
```

---

## ⚙️ Configuration

All experiments are driven by YAML/JSON config files in `configs/`. This ensures reproducibility and makes it easy to share experimental setups.

### Available Configs

| Task | Config File | Description |
|------|------------|-------------|
| Layout Training | `train_layout_d3pm_masked_sparse_80k.yaml` | Train Stage A with 80k steps |
| Ratio Prior | `compute_ratio_prior_loveda_train.yaml` | Compute class statistics |
| ControlNet Training | `train_controlnet_ratio_loveda_1024.yaml` | Train Stage B at 1024px |
| Sampling | `sample_pair_ckpt40000_building0.4.yaml` | Generate with 40% buildings |

### Config Tips

- 📝 All config examples are in `configs/`
- 🔄 To resume training: set `resume_from_checkpoint: "checkpoint-XXXXX"` in your config
- 🎯 Dataset paths and domains are centralized in configs—edit once, reuse everywhere
- 🔧 CLI arguments override config values for quick experiments

---

## 📊 Outputs

### Training Outputs

Checkpoints include:
- `training_config.json` - Complete training configuration
- `class_names.json` - Class label mappings
- Model weights and optimizer states

### Sampling Outputs

Each generated sample produces:
- `image.png` - Synthetic RGB image
- `layout.png` - Corresponding semantic layout
- `metadata.json` - Generation parameters and class ratios

---

## 📦 Pre-Generated Datasets

We provide synthetic datasets used in our paper:

🔗 **[Download from Google Drive](https://drive.google.com/drive/folders/14cMpLTgvcLdXhRY0kGhFKpDRMvpok90h?usp=sharing)**

These datasets demonstrate SyntheticGen's ability to generate balanced, high-quality remote sensing imagery for long-tail class mitigation.

---

## 📄 Citation

If you find SyntheticGen useful in your research, please consider citing:

```bibtex
@article{syntheticgen2024,
  title={Mitigating Long-Tail Bias in LoveDA via Prompt-Controlled Diffusion Augmentation},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- LoveDA dataset creators for providing high-quality annotated remote sensing data
- The Hugging Face Diffusers team for excellent diffusion model infrastructure
- ControlNet authors for the controllable generation framework

---

<div align="center">

[Report Bug](https://github.com/yourusername/syntheticgen/issues) · [Request Feature](https://github.com/yourusername/syntheticgen/issues) · [Paper](https://arxiv.org/abs/your-paper-id)

</div>
```