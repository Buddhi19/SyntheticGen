<div align="center">

# 🎨 SyntheticGen

### Mitigating Long-Tail Bias via Prompt-Controlled Diffusion Augmentation

*Addressing class imbalance in remote sensing datasets through controlled synthetic generation*

[![Accepted at IEEE IGARSS 2026](https://img.shields.io/badge/Accepted-IEEE%20IGARSS%202026-1f77b4)](#)
[![arXiv Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2602.04749)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-blue)](https://drive.google.com/drive/folders/14cMpLTgvcLdXhRY0kGhFKpDRMvpok90h?usp=sharing)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 🌟 Overview

**SyntheticGen** tackles the long-tail distribution problem in LoveDA by generating synthetic imagery with *explicit control* over class ratios. You can specify exactly what proportion of each land cover class should appear in the output.

## 🔥🔥 Updates
- Our paper was accepted to EEE International Geoscience and Remote Sensing Symposium (IGARSS) 2026.

### ✨ Highlights
- Two-stage pipeline: ratio-conditioned layout D3PM + ControlNet image synthesis.
- Full or sparse ratio control (e.g., `building:0.4`).
- Config-first workflow for reproducible experiments.

<div align="center">
  <img src="docs/results.png" alt="SyntheticGen Results" width="100%">
</div>


---

## ❓ What we try to answer

🛰️ **Why is remote-sensing segmentation still difficult, even with strong modern models?**  


Because the problem is not only in the model — it is also in the data. Some land-cover classes appear again and again, while others are so rare that the model barely gets a chance to learn them. In LoveDA, this becomes even more challenging because the dataset is split into **Urban** and **Rural** domains, each with different scene characteristics and different class distributions.

⚖️ **So what if we could control the data instead of just accepting it as it is?**  


That is exactly the idea behind **SyntheticGen**. Instead of using augmentation as a random process, SyntheticGen makes it **controllable**. Users can explicitly specify target class ratios and domain conditions during generation, making it possible to create synthetic samples that are not just more numerous, but more *useful*. This means rare classes can be strengthened deliberately, while still preserving realistic layouts and domain-consistent appearance.

🧠 **What makes SyntheticGen stand out?**  


Its strength lies in a carefully designed **two-stage pipeline**. First, a **ratio-conditioned discrete diffusion model** generates semantically meaningful layouts. Then, a **ControlNet-guided image synthesis stage** converts those layouts into realistic remote-sensing imagery. By separating **semantic control** from **visual rendering**, the framework achieves something highly valuable: it is both **principled** and **practical**.

🚀 **Why does that matter beyond this single benchmark?**  


Because this is not just another generative model for remote sensing. SyntheticGen introduces a targeted augmentation strategy for improving segmentation under **class imbalance** and **domain shift**, and shows that synthetic data can be used not just to add more images, but to add the **right images**.

🌍 **The bigger message**  


SyntheticGen is a step toward **data-centric remote-sensing segmentation** — a setting where the training distribution is no longer passively accepted, but actively designed. Our paper shows that better segmentation is not only about building better models, but also about building better data.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Buddhi19/SyntheticGen.git
cd SyntheticGen
pip install -r requirements.txt
```

### Generate Your First Synthetic Image
```bash
python src/scripts/sample_pair.py \
  --config configs/sample_pair_ckpt40000_building0.4.yaml
```

---

## 📚 Usage

### Training Pipeline (Configs)

**Stage A: Train Layout Generator (D3PM)**
```bash
python src/scripts/train_layout_d3pm.py \
  --config configs/train_layout_d3pm_masked_sparse_80k.yaml
```

**(Optional) Ratio Prior for Sparse Conditioning**
```bash
python src/scripts/compute_ratio_prior.py \
  --config configs/compute_ratio_prior_loveda_train.yaml
```

**Stage B: Train Image Generator (ControlNet)**
```bash
python src/scripts/train_controlnet_ratio.py \
  --config configs/train_controlnet_ratio_loveda_1024.yaml
```

### Inference / Sampling (Configs)

**End-to-end sampling (layout -> image):**
```bash
python src/scripts/sample_pair.py \
  --config configs/sample_pair_ckpt40000_building0.4.yaml
```

**Override config parameters via CLI if needed:**
```bash
python src/scripts/sample_pair.py \
  --config configs/sample_pair_ckpt40000_building0.4.yaml \
  --ratios "building:0.4,forest:0.3" \
  --save_dir outputs/custom_generation
```

---

## ⚙️ Configuration

All experiments are driven by YAML/JSON config files in `configs/`.

| Task | Script | Example Config |
|------|--------|----------------|
| Layout Training | `src/scripts/train_layout_d3pm.py` | `configs/train_layout_d3pm_masked_sparse_80k.yaml` |
| Ratio Prior | `src/scripts/compute_ratio_prior.py` | `configs/compute_ratio_prior_loveda_train.yaml` |
| ControlNet Training | `src/scripts/train_controlnet_ratio.py` | `configs/train_controlnet_ratio_loveda_1024.yaml` |
| Sampling / Inference | `src/scripts/sample_pair.py` | `configs/sample_pair_ckpt40000_building0.4.yaml` |

**Config tips**
- Examples live in `configs/`.
- To resume training, set `resume_from_checkpoint: "checkpoint-XXXXX"` in your config.
- Dataset roots and domains are centralized in configs; edit once, reuse everywhere.
- CLI flags override config values for quick experiments.

---

## 📁 Data Format

### LoveDA Dataset Structure
```
LoveDA/
  Train/
    Train/            # some releases include this extra nesting
      Urban/
        images_png/
        masks_png/
      Rural/
        images_png/
        masks_png/
    Urban/
      images_png/
      masks_png/
    Rural/
      images_png/
      masks_png/
  Val/
    ...
```

### Generic Dataset Structure
```
your_dataset/
  images/
    image_001.png
  masks/
    image_001.png   # label map with matching stem
```

---

## 📦 Pre-Generated Datasets

We provide synthetic datasets used in the paper:
https://drive.google.com/drive/folders/14cMpLTgvcLdXhRY0kGhFKpDRMvpok90h?usp=sharing

---

## 🧾 Outputs
- Checkpoints include `training_config.json` and `class_names.json`.
- Sampling writes `image.png`, `layout.png`, and `metadata.json`.

---

## 📄 Citation
```bibtex
@misc{wijenayake2026mitigating,
      title={Mitigating Long-Tail Bias in LoveDA via Prompt-Controlled Diffusion Augmentation},
      author={Buddhi Wijenayake and Nichula Wasalathilake and Roshan Godaliyadda and Vijitha Herath and Parakrama Ekanayake and Vishal M. Patel},
      year={2026},
      eprint={2602.04749},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.04749}
}
```

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments
- LoveDA dataset creators for high-quality annotated remote sensing data
- Hugging Face Diffusers for diffusion model infrastructure
- ControlNet authors for controllable generation

---

<div align="center">

[Report Bug](https://github.com/Buddhi19/SyntheticGen/issues) · [Request Feature](https://github.com/Buddhi19/SyntheticGen/issues)

</div>
