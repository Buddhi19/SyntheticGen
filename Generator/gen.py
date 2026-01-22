from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


CLASS_NAMES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
IGNORE_INDEX = 255
EPS = 1e-12
MASK_EXTS = {".png", ".tif", ".tiff"}
MASK_RGB_DIRNAME = "mask_rgb_png"
LOVEDA_PALETTE = np.array(
    [
        [255, 255, 255],  # background
        [255, 0, 0],      # building
        [255, 255, 0],    # road
        [0, 0, 255],      # water
        [159, 129, 183],  # barren
        [0, 255, 0],      # forest
        [255, 195, 128],  # agriculture
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate balanced synthetic LoveDA pairs.")
    parser.add_argument("--real_root", type=str, default="/data/inr/llm/Datasets/LOVEDA/Train/Train", help="LoveDA root or split root.")
    parser.add_argument("--real_split", type=str, default="Train", help="Split name when real_root is dataset root.")
    parser.add_argument(
        "--real_mask_format",
        type=str,
        default="auto",
        choices=["auto", "loveda_raw", "indexed"],
        help="Mask format for real data.",
    )
    parser.add_argument("--synth_root", type=str, default=None, help="Synthetic dataset root.")
    parser.add_argument("--synth_total", type=int, default=2000, help="Total synthetic samples to generate.")

    parser.add_argument("--layout_ckpt", type=str, default=None, help="Layout DDPM/D3PM checkpoint directory.")
    parser.add_argument("--layout_checkpoint", type=int, default=79000, help="Layout checkpoint step.")
    parser.add_argument("--layout_diffusion_type", type=str, default="d3pm", choices=["d3pm", "ddpm"])
    parser.add_argument("--domain_cond_scale", type=float, default=1.0)
    parser.add_argument("--controlnet_ckpt", type=str, default=None, help="ControlNet checkpoint directory.")
    parser.add_argument("--base_model", type=str, default="/home/nvidia/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14", help="Base SD model path.")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--num_steps_layout", type=int, default=200)
    parser.add_argument("--num_steps_image", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--control_scale", type=float, default=1.0)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm", "dpmpp_2m"])
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size per sample_pair call.")
    parser.add_argument("--cuda_visible_devices", type=str, default="7")
    parser.add_argument("--pytorch_cuda_alloc_conf", type=str, default="expandable_segments:True")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index for multi-process runs.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total shards for multi-process runs.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by scanning existing synthetic masks and only generating remaining quota.",
    )
    parser.add_argument(
        "--only_domain",
        type=str,
        default="both",
        choices=["both", "rural", "urban"],
        help="Restrict generation to a single domain.",
    )
    parser.add_argument("--run_id", type=str, default=None, help="Optional run identifier for logs/tmp.")
    parser.add_argument("--log_path", type=str, default=None, help="Override generation log path.")
    parser.add_argument("--manifest_path", type=str, default=None, help="Override manifest path.")
    parser.add_argument("--tmp_root", type=str, default=None, help="Override temp output root.")

    parser.add_argument("--beta", type=float, default=0.55, help="Flattening power for non-background classes.")
    parser.add_argument("--gamma", type=float, default=0.7, help="Rarity boost power.")
    parser.add_argument(
        "--target_mode",
        type=str,
        default="uniform_nonbg",
        choices=["uniform_nonbg", "custom_json"],
        help="How to define desired per-image ratios.",
    )
    parser.add_argument(
        "--target_bg",
        type=float,
        default=0.20,
        help="Desired background ratio per image (lower => minorities higher).",
    )
    parser.add_argument(
        "--target_json",
        type=str,
        default=None,
        help="Path to JSON with per-domain target ratios (custom_json).",
    )
    parser.add_argument(
        "--ratios_topk",
        type=int,
        default=3,
        help="How many non-background classes to constrain per sample (in addition to background).",
    )
    parser.add_argument(
        "--accept_ref",
        type=str,
        default="target",
        choices=["target", "real", "none"],
        help="Acceptance reference distribution for global/KL checks.",
    )
    parser.add_argument("--urban_building_min", type=float, default=0.20)
    parser.add_argument("--rural_building_min", type=float, default=0.10)
    parser.add_argument("--min_ratio", type=float, default=0.02)
    parser.add_argument("--max_ratio", type=float, default=0.60)
    parser.add_argument("--max_delta_json", type=str, default=None, help="JSON path for per-class delta caps.")
    parser.add_argument("--focus_abs_tol", type=float, default=0.14)
    parser.add_argument("--global_max_abs", type=float, default=0.42)
    parser.add_argument("--kl_max", type=float, default=1.30)
    parser.add_argument("--max_attempts", type=int, default=12)
    parser.add_argument(
        "--prompt_rural",
        type=str,
        default="A high-resolution satellite image of a rural area",
    )
    parser.add_argument(
        "--prompt_urban",
        type=str,
        default="A high-resolution satellite image of an urban area",
    )

    return parser.parse_args()


def find_repo_root() -> Path:
    script_root = Path(__file__).resolve().parents[1]
    candidate = script_root / "src" / "scripts" / "sample_pair.py"
    if candidate.is_file():
        return script_root

    cwd = Path.cwd().resolve()
    for base in [cwd] + list(cwd.parents):
        candidate = base / "src" / "scripts" / "sample_pair.py"
        if candidate.is_file():
            return base
        candidate = base / "SyntheticGen" / "src" / "scripts" / "sample_pair.py"
        if candidate.is_file():
            return base / "SyntheticGen"

    raise FileNotFoundError("Could not locate src/scripts/sample_pair.py from current path parents.")


def _find_dir(root: Path, candidates: Iterable[str]) -> Optional[Path]:
    for name in candidates:
        p = root / name
        if p.is_dir():
            return p
    return None


def _resolve_split_dir(root: Path, split: str) -> Optional[Path]:
    direct = root / split
    if direct.is_dir():
        nested = direct / split
        if nested.is_dir():
            return nested
        return direct
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.lower() == split.lower():
            nested = entry / entry.name
            if nested.is_dir():
                return nested
            return entry
    return None


def _resolve_domain_dir(split_dir: Path, domain: str) -> Optional[Path]:
    direct = split_dir / domain
    if direct.is_dir():
        return direct
    for entry in split_dir.iterdir():
        if entry.is_dir() and entry.name.lower() == domain.lower():
            return entry
    for entry in split_dir.iterdir():
        if not entry.is_dir():
            continue
        nested = entry / domain
        if nested.is_dir():
            return nested
        for subentry in entry.iterdir():
            if subentry.is_dir() and subentry.name.lower() == domain.lower():
                return subentry
    return None


def _default_loveda_root() -> Optional[Path]:
    env_root = os.environ.get("LOVEDA_ROOT") or os.environ.get("LOVE_DA_ROOT")
    if env_root:
        p = Path(env_root).expanduser()
        if p.is_dir():
            return p

    script_root = Path(__file__).resolve()
    for base in [script_root] + list(script_root.parents) + [Path.cwd().resolve()]:
        candidate = base / "Datasets" / "LOVEDA"
        if candidate.is_dir():
            return candidate

    candidate = Path("/data/inr/llm/Datasets/LOVEDA")
    if candidate.is_dir():
        return candidate
    return None


def resolve_real_root(real_root: Optional[str], split: str) -> Path:
    if real_root:
        root = Path(real_root).expanduser().resolve()
    else:
        found = _default_loveda_root()
        if found is None:
            raise FileNotFoundError("Could not locate LoveDA root. Pass --real_root explicitly.")
        root = found

    for domain in ("Rural", "Urban"):
        if _resolve_domain_dir(root, domain) is None:
            split_dir = _resolve_split_dir(root, split)
            if split_dir is None:
                raise FileNotFoundError(f"Could not resolve split {split} under {root}")
            return split_dir
    return root


def load_mask_as_indexed(mask_path: Path, mask_format: str) -> np.ndarray:
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    if mask_format == "indexed":
        return arr.astype(np.int64)
    if mask_format == "loveda_raw":
        out = arr.astype(np.int64)
        ignore = out == 0
        out[ignore] = IGNORE_INDEX
        out[~ignore] = out[~ignore] - 1
        return out.astype(np.int64)

    # auto
    uniq = np.unique(arr)
    if 255 in uniq:
        return arr.astype(np.int64)
    mx = int(arr.max())
    mn = int(arr.min())
    if mx <= 7 and mn >= 0 and 0 in uniq:
        out = arr.astype(np.int64)
        ignore = out == 0
        out[ignore] = IGNORE_INDEX
        out[~ignore] = out[~ignore] - 1
        return out.astype(np.int64)
    return arr.astype(np.int64)


def hist_counts(mask_idx: np.ndarray) -> np.ndarray:
    flat = mask_idx.reshape(-1)
    flat = flat[(flat != IGNORE_INDEX) & (flat >= 0) & (flat < len(CLASS_NAMES))]
    return np.bincount(flat, minlength=len(CLASS_NAMES)).astype(np.float64)


def safe_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    q = np.clip(q, eps, 1.0)
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def domain_dirs(root: Path, domain_title: str) -> Tuple[Path, Path]:
    droot = _resolve_domain_dir(root, domain_title)
    if droot is None:
        raise FileNotFoundError(f"Domain dir not found for {domain_title} under {root}")
    masks = _find_dir(droot, ["masks_png", "masks", "labels", "SegmentationClass"])
    imgs = _find_dir(droot, ["images_png", "images", "imgs", "JPEGImages"])
    if masks is None:
        raise FileNotFoundError(f"No mask dir found for {domain_title} under {droot}")
    if imgs is None:
        raise FileNotFoundError(f"No image dir found for {domain_title} under {droot}")
    return imgs, masks


def _list_masks(masks_dir: Path) -> List[Path]:
    return sorted([p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() in MASK_EXTS])


def scan_domain(root: Path, domain_title: str, mask_format: str) -> Dict[str, object]:
    imgs_dir, masks_dir = domain_dirs(root, domain_title)
    mask_files = _list_masks(masks_dir)
    if not mask_files:
        raise FileNotFoundError(f"No masks found for {domain_title} under {masks_dir}")
    totals = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    valid_px_total = 0.0

    for mp in mask_files:
        m = load_mask_as_indexed(mp, mask_format)
        c = hist_counts(m)
        totals += c
        valid_px_total += float(c.sum())

    return {
        "n": len(mask_files),
        "totals": totals,
        "valid_px_total": valid_px_total,
        "avg_valid_px": valid_px_total / max(1, len(mask_files)),
        "masks_dir": str(masks_dir),
        "imgs_dir": str(imgs_dir),
    }


def scan_synth_domain(synth_root: Path, domain_title: str, mask_format: str) -> Dict[str, object]:
    split_root = synth_root / "Train" / "Train"
    if not split_root.is_dir():
        return {
            "n": 0,
            "totals": np.zeros(len(CLASS_NAMES), dtype=np.float64),
            "valid_px_total": 0.0,
            "avg_valid_px": 0.0,
        }
    try:
        _, masks_dir = domain_dirs(split_root, domain_title)
    except FileNotFoundError:
        return {
            "n": 0,
            "totals": np.zeros(len(CLASS_NAMES), dtype=np.float64),
            "valid_px_total": 0.0,
            "avg_valid_px": 0.0,
        }
    mask_files = _list_masks(masks_dir)
    if not mask_files:
        return {
            "n": 0,
            "totals": np.zeros(len(CLASS_NAMES), dtype=np.float64),
            "valid_px_total": 0.0,
            "avg_valid_px": 0.0,
        }
    totals = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    valid_px_total = 0.0
    for mp in mask_files:
        m = load_mask_as_indexed(mp, mask_format)
        c = hist_counts(m)
        totals += c
        valid_px_total += float(c.sum())
    return {
        "n": len(mask_files),
        "totals": totals,
        "valid_px_total": valid_px_total,
        "avg_valid_px": valid_px_total / max(1, len(mask_files)),
    }


def compute_domain_quota(nr: int, nu: int, synth_total: int) -> Dict[str, int]:
    target_per_domain = math.floor((nr + nu + synth_total) / 2.0)
    need_r = max(0, target_per_domain - nr)
    need_u = max(0, target_per_domain - nu)
    allocated = need_r + need_u
    rem = synth_total - allocated
    if rem > 0:
        need_r += rem // 2
        need_u += rem - rem // 2

    while (need_r + need_u) > synth_total:
        if need_r >= need_u and need_r > 0:
            need_r -= 1
        elif need_u > 0:
            need_u -= 1
        else:
            break

    return {"rural": int(need_r), "urban": int(need_u)}


def split_quota_for_shard(total: int, num_shards: int, shard_id: int) -> int:
    if num_shards <= 1:
        return int(total)
    base = int(total) // int(num_shards)
    rem = int(total) % int(num_shards)
    return base + (1 if shard_id < rem else 0)


def ensure_synth_structure(root: Path) -> None:
    for dom in ["Rural", "Urban"]:
        (root / "Train" / "Train" / dom / "images_png").mkdir(parents=True, exist_ok=True)
        (root / "Train" / "Train" / dom / "masks_png").mkdir(parents=True, exist_ok=True)
        (root / "Train" / "Train" / dom / MASK_RGB_DIRNAME).mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "tmp").mkdir(parents=True, exist_ok=True)


def save_mask_raw_from_indexed(mask_idx: np.ndarray, out_path: Path) -> None:
    raw = np.zeros_like(mask_idx, dtype=np.uint8)
    ignore = mask_idx == IGNORE_INDEX
    raw[ignore] = 0
    raw[~ignore] = (mask_idx[~ignore] + 1).astype(np.uint8)
    Image.fromarray(raw, mode="L").save(out_path)


def colorize_mask(mask_idx: np.ndarray) -> np.ndarray:
    safe = mask_idx.copy()
    safe[safe == IGNORE_INDEX] = 0
    safe = np.clip(safe, 0, len(LOVEDA_PALETTE) - 1)
    return LOVEDA_PALETTE[safe]


def load_targets_from_json(path: str) -> Dict[str, np.ndarray]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # expected: {"rural":{"background":0.2,...}, "urban":{...}}
    out = {}
    for dom in ["rural", "urban"]:
        v = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        for i, name in enumerate(CLASS_NAMES):
            v[i] = float(data[dom][name])
        v = np.clip(v, EPS, 1.0)
        v = v / v.sum()
        out[dom] = v
    return out


def compute_target(
    domain: str,
    args: argparse.Namespace,
    cached_custom: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    if args.target_mode == "custom_json":
        if cached_custom is None:
            raise ValueError("--target_mode custom_json requires --target_json")
        return cached_custom[domain].copy()

    # uniform over non-background with fixed background mass
    t = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    bg = float(np.clip(args.target_bg, 0.05, 0.80))
    t[0] = bg
    rest = (1.0 - bg) / (len(CLASS_NAMES) - 1)
    t[1:] = rest
    return t


def _renormalize_requests(
    req: Dict[int, float],
    fixed_keys: Optional[Iterable[int]] = None,
    min_bg: float = 0.05,
) -> Dict[int, float]:
    # Ensure requested ratios are feasible; keep fixed keys at requested values.
    fixed = set(int(k) for k in fixed_keys) if fixed_keys else set()
    fixed_sum = sum(req[k] for k in req.keys() if k in fixed)
    variable_keys = [k for k in req.keys() if k not in fixed]
    if not variable_keys:
        return req

    remaining = 1.0 - min_bg - fixed_sum
    if remaining <= 0:
        for k in variable_keys:
            req[k] = 0.0
        return req

    var_sum = sum(req[k] for k in variable_keys)
    if var_sum <= remaining:
        return req

    scale = remaining / max(EPS, var_sum)
    for k in variable_keys:
        req[k] = float(req[k] * scale)
    return req


def build_ratios_str(
    domain: str,
    cur_counts: np.ndarray,
    plan_total: float,
    t: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[str, Dict[int, float]]:
    """
    Always constrain background to target_bg, and constrain top-K deficit non-bg classes.
    Returns: ratios_str and a dict {class_index: requested_ratio}.
    """
    pi = cur_counts / max(EPS, cur_counts.sum())
    desired = plan_total * t
    deficit = np.maximum(0.0, desired - cur_counts)  # (K,)

    # Always include background constraint (key for making minorities high)
    req: Dict[int, float] = {0: float(np.clip(t[0], args.min_ratio, args.max_ratio))}
    fixed_keys = {0}
    if domain == "urban":
        bmin = float(getattr(args, "urban_building_min", 0.20))
        bmin = float(np.clip(bmin, args.min_ratio, args.max_ratio))
        base = float(np.clip(max(bmin, float(t[1])), args.min_ratio, args.max_ratio))
        span = max(0.0, float(args.max_ratio) - base)
        if span > 0:
            jitter = min(0.15, span)
            base = float(base + np.random.uniform(0.0, jitter))
        req[1] = base
        fixed_keys = {0, 1}
    elif domain == "rural":
        bmin = float(getattr(args, "rural_building_min", 0.10))
        if bmin > 0.0:
            bmin = float(np.clip(bmin, args.min_ratio, args.max_ratio))
            req[1] = float(np.clip(max(bmin, float(t[1])), args.min_ratio, args.max_ratio))
            fixed_keys = {0, 1}

    # Rank non-background by deficit (optionally with rarity boost)
    score = deficit.copy()
    score[0] = 0.0
    if 1 in fixed_keys:
        score[1] = 0.0
    if args.gamma > 0:
        score *= np.power(np.clip(pi, EPS, 1.0), -float(args.gamma))
    order = np.argsort(score)[::-1]
    topk = int(args.ratios_topk)
    if 1 in fixed_keys:
        topk = max(1, topk - 1)
    chosen = [k for k in order if k not in fixed_keys and score[k] > 0][:topk]
    if not chosen:
        # fallback: pick smallest-frequency non-bg
        nonbg = np.arange(1, len(CLASS_NAMES))
        nonbg = np.array([k for k in nonbg if k not in fixed_keys], dtype=int)
        if len(nonbg) > 0:
            chosen = [int(nonbg[np.argmin(pi[nonbg])])]

    for k in chosen:
        # push class toward target ratio (not the real prior)
        r = float(t[k])
        # small stochasticity helps exploration
        r = float(np.clip(r + np.random.normal(0.0, 0.01), args.min_ratio, args.max_ratio))
        req[int(k)] = r

    # Make sure constraints are feasible
    req = _renormalize_requests(req, fixed_keys=fixed_keys, min_bg=0.05)

    ratios_str = ",".join([f"{CLASS_NAMES[k]}:{req[k]:.4f}" for k in sorted(req.keys())])
    return ratios_str, req


def pick_focus_class(deficit: np.ndarray, pi: np.ndarray, gamma: float) -> int:
    rarity = np.power(np.clip(pi, EPS, 1.0), -gamma)
    score = deficit * rarity
    score[0] = 0.0
    s = score.sum()
    if s <= 0:
        w = np.power(np.clip(pi, EPS, 1.0), -1.0)
        w[0] = 0.0
        w = w / w.sum()
        return int(np.random.choice(np.arange(len(CLASS_NAMES)), p=w))
    p = score / s
    return int(np.random.choice(np.arange(len(CLASS_NAMES)), p=p))


def propose_ratio(
    domain: str,
    focus_k: int,
    t: np.ndarray,
    pi: np.ndarray,
    mu_real: Dict[str, np.ndarray],
    max_delta: Dict[str, float],
) -> float:
    mu0 = float(mu_real[domain][focus_k])
    if mu0 <= 0:
        mu0 = float(pi[focus_k])
    desired_lift = float(t[focus_k] / max(pi[focus_k], EPS))
    name = CLASS_NAMES[focus_k]
    cap = float(max_delta.get(name, 0.7))
    delta = min(cap, max(0.0, 0.85 * (desired_lift - 1.0)))
    r = mu0 * (1.0 + delta)
    r = float(r + np.random.normal(0.0, 0.015))
    r = float(np.clip(r, 0.01, 0.60))
    r = float(np.clip(r, mu0 * (1.0 - 0.25), mu0 * (1.0 + cap)))
    return r


def accept_layout(
    domain: str,
    p: np.ndarray,
    req: Dict[int, float],
    mu_real: Dict[str, np.ndarray],
    t: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, float]]:
    # (i) must satisfy each requested class ratio within tolerance
    building_min = False
    if domain == "urban":
        building_min = True
    elif domain == "rural" and float(getattr(args, "rural_building_min", 0.0)) > 0.0:
        building_min = True

    if building_min and 1 in req:
        if float(p[1]) < float(req[1]):
            return False, {"building_min": float(req[1]), "building_p": float(p[1])}

    max_req_err = 0.0
    for k, r in req.items():
        if building_min and k == 1:
            continue
        max_req_err = max(max_req_err, float(abs(p[k] - r)))
    if max_req_err > float(args.focus_abs_tol):
        return False, {"max_req_err": max_req_err}

    # (ii) optional global plausibility check
    if args.accept_ref == "none":
        return True, {"max_req_err": max_req_err}

    ref = mu_real[domain] if args.accept_ref == "real" else t
    max_abs = float(np.max(np.abs(p - ref)))
    kl = safe_kl(p, ref)

    ok = (max_abs <= float(args.global_max_abs)) and (kl <= float(args.kl_max))
    return ok, {"max_req_err": max_req_err, "max_abs": max_abs, "kl": kl}


def run_sample_pair(
    sample_ctx: object,
    sample_pair_module: object,
    sample_args: argparse.Namespace,
    tmp_out: Path,
    domain: str,
    ratios_str: str,
    seed: int,
    prompt: str,
) -> None:
    per_call = copy.copy(sample_args)
    per_call.save_dir = str(tmp_out)
    per_call.domain = str(domain)
    per_call.ratios = ratios_str
    per_call.ratios_json = None
    per_call.prompt = prompt
    per_call.seed = int(seed)
    sample_pair_module.generate_with_context(sample_ctx, per_call)


def write_synth_sample(
    domain: str,
    stem: str,
    image_path: Path,
    layout_path: Path,
    layout_color_path: Optional[Path],
    synth_root: Path,
    metadata: Optional[Dict[str, object]],
) -> Tuple[Path, Path, Path, Path, np.ndarray]:
    dom_title = "Rural" if domain == "rural" else "Urban"

    img_dst = synth_root / "Train" / "Train" / dom_title / "images_png" / f"{stem}.png"
    msk_dst = synth_root / "Train" / "Train" / dom_title / "masks_png" / f"{stem}.png"
    rgb_dst = synth_root / "Train" / "Train" / dom_title / MASK_RGB_DIRNAME / f"{stem}.png"
    meta_dst = synth_root / "meta" / f"{stem}.json"

    shutil.move(str(image_path), str(img_dst))

    m_idx = np.array(Image.open(layout_path))
    if m_idx.ndim == 3:
        m_idx = m_idx[:, :, 0]
    m_idx = m_idx.astype(np.int64)
    save_mask_raw_from_indexed(m_idx, msk_dst)

    if layout_color_path is not None and layout_color_path.exists():
        shutil.move(str(layout_color_path), str(rgb_dst))
    else:
        rgb = colorize_mask(m_idx)
        Image.fromarray(rgb, mode="RGB").save(rgb_dst)

    meta_payload: Dict[str, object] = dict(metadata) if isinstance(metadata, dict) else {"domain": domain}
    meta_payload["domain"] = domain
    meta_payload["image_path"] = str(img_dst)
    meta_payload["layout_path"] = str(msk_dst)
    meta_payload["layout_color_path"] = str(rgb_dst)
    meta_dst.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    return img_dst, msk_dst, rgb_dst, meta_dst, m_idx


def _next_index(images_dir: Path, domain: str) -> int:
    if not images_dir.is_dir():
        return 0
    pattern = re.compile(rf"^{re.escape(domain)}_(\d+)$")
    max_idx = -1
    for path in images_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        match = pattern.match(path.stem)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def _resolve_output_path(tmp_out: Path, path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = tmp_out / p
    return p


def load_batch_outputs(tmp_out: Path) -> List[Dict[str, object]]:
    meta_path = tmp_out / "metadata.json"
    outputs: List[Dict[str, object]] = []
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else [data]
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            layout_path = entry.get("layout_path")
            image_path = entry.get("image_path")
            if not layout_path or not image_path:
                continue
            outputs.append(
                {
                    "layout_path": _resolve_output_path(tmp_out, str(layout_path)),
                    "image_path": _resolve_output_path(tmp_out, str(image_path)),
                    "layout_color_path": _resolve_output_path(tmp_out, str(entry["layout_color_path"]))
                    if entry.get("layout_color_path")
                    else None,
                    "sample_index": entry.get("sample_index", idx),
                    "metadata": entry,
                }
            )
        if outputs:
            return outputs

    layouts = sorted([p for p in tmp_out.glob("layout*.png") if "layout_color" not in p.stem])
    for idx, layout_path in enumerate(layouts):
        suffix = layout_path.stem[len("layout") :]
        image_path = tmp_out / f"image{suffix}.png"
        color_path = tmp_out / f"layout_color{suffix}.png"
        outputs.append(
            {
                "layout_path": layout_path,
                "image_path": image_path,
                "layout_color_path": color_path if color_path.exists() else None,
                "sample_index": idx,
                "metadata": None,
            }
        )
    return outputs


def main() -> None:
    args = parse_args()

    cached_custom_targets = None
    if args.target_mode == "custom_json":
        if not args.target_json:
            raise ValueError("--target_json is required when --target_mode custom_json")
        cached_custom_targets = load_targets_from_json(args.target_json)

    if args.layout_ckpt is None or args.controlnet_ckpt is None or args.base_model is None:
        raise ValueError("Required: --layout_ckpt, --controlnet_ckpt, and --base_model.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1.")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards.")

    base_seed = int(args.seed) + int(args.shard_id) * 1000003
    random.seed(base_seed)
    np.random.seed(base_seed)

    repo_root = find_repo_root()
    sample_pair_path = repo_root / "src" / "scripts" / "sample_pair.py"
    if not sample_pair_path.is_file():
        raise FileNotFoundError(f"sample_pair.py not found: {sample_pair_path}")

    real_root = resolve_real_root(args.real_root, args.real_split)
    real_r = scan_domain(real_root, "Rural", args.real_mask_format)
    real_u = scan_domain(real_root, "Urban", args.real_mask_format)

    mu_real = {
        "rural": real_r["totals"] / max(1.0, float(real_r["totals"].sum())),
        "urban": real_u["totals"] / max(1.0, float(real_u["totals"].sum())),
    }

    domain_quota_total = compute_domain_quota(int(real_r["n"]), int(real_u["n"]), int(args.synth_total))

    synth_root = Path(args.synth_root).expanduser().resolve() if args.synth_root else (repo_root.parent / "SyntheticDataset")
    ensure_synth_structure(synth_root)

    existing = {"rural": 0, "urban": 0}
    synth_totals = {
        "rural": np.zeros(len(CLASS_NAMES), dtype=np.float64),
        "urban": np.zeros(len(CLASS_NAMES), dtype=np.float64),
    }
    if args.resume:
        synth_r = scan_synth_domain(synth_root, "Rural", "loveda_raw")
        synth_u = scan_synth_domain(synth_root, "Urban", "loveda_raw")
        existing = {"rural": int(synth_r["n"]), "urban": int(synth_u["n"])}
        synth_totals = {"rural": synth_r["totals"], "urban": synth_u["totals"]}

    domain_quota_remaining = {
        "rural": max(0, domain_quota_total["rural"] - existing["rural"]),
        "urban": max(0, domain_quota_total["urban"] - existing["urban"]),
    }
    if args.only_domain in ("rural", "urban"):
        other = "urban" if args.only_domain == "rural" else "rural"
        domain_quota_remaining[other] = 0

    domain_quota = {
        "rural": split_quota_for_shard(domain_quota_remaining["rural"], args.num_shards, args.shard_id),
        "urban": split_quota_for_shard(domain_quota_remaining["urban"], args.num_shards, args.shard_id),
    }

    tmp_root = (
        Path(args.tmp_root).expanduser().resolve()
        if args.tmp_root
        else (synth_root / "tmp" / f"run_{args.run_id}" if args.run_id else synth_root / "tmp")
    )
    tmp_root.mkdir(parents=True, exist_ok=True)

    start_idx = {
        "rural": _next_index(synth_root / "Train" / "Train" / "Rural" / "images_png", "rural"),
        "urban": _next_index(synth_root / "Train" / "Train" / "Urban" / "images_png", "urban"),
    }
    if start_idx["rural"] > 0 or start_idx["urban"] > 0:
        print(f"Warning: existing synthetic images found, starting at {start_idx}")

    max_delta = {
        "background": 0.15,
        "building": 1.20,
        "road": 1.10,
        "water": 0.80,
        "barren": 1.00,
        "forest": 0.35,
        "agriculture": 0.35,
    }
    if args.max_delta_json:
        max_delta.update(json.loads(Path(args.max_delta_json).read_text(encoding="utf-8")))

    cur = {
        "rural": real_r["totals"].copy() + synth_totals["rural"],
        "urban": real_u["totals"].copy() + synth_totals["urban"],
    }
    avg_valid_px = {
        "rural": float(real_r["avg_valid_px"]),
        "urban": float(real_u["avg_valid_px"]),
    }

    accepted = {"rural": 0, "urban": 0}
    manifest: List[Dict[str, object]] = []

    log_path = (
        Path(args.log_path).expanduser().resolve()
        if args.log_path
        else synth_root / "meta" / (f"generation_log_{args.run_id}.jsonl" if args.run_id else "generation_log.jsonl")
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if args.pytorch_cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", str(args.pytorch_cuda_alloc_conf))

    def choose_domain() -> str:
        if args.only_domain in ("rural", "urban"):
            return args.only_domain
        rem_r = domain_quota["rural"] - accepted["rural"]
        rem_u = domain_quota["urban"] - accepted["urban"]
        if rem_r <= 0 and rem_u <= 0:
            return "rural"
        if rem_r <= 0:
            return "urban"
        if rem_u <= 0:
            return "rural"
        pr = rem_r / (rem_r + rem_u)
        return "rural" if random.random() < pr else "urban"

    total_to_make = sum(domain_quota.values())
    print("Real counts:", {"rural": real_r["n"], "urban": real_u["n"]})
    if args.resume:
        print("Existing synthetic counts:", existing)
        print("Synthetic domain quota (remaining):", domain_quota_remaining, "sum=", sum(domain_quota_remaining.values()))
        if args.num_shards > 1:
            print("Synthetic domain quota (this shard):", domain_quota, "sum=", total_to_make)
        else:
            print("Synthetic domain quota (this run):", domain_quota, "sum=", total_to_make)
    else:
        if args.num_shards > 1:
            print("Synthetic domain quota (this shard):", domain_quota, "sum=", total_to_make)
            print("Synthetic domain quota (all shards):", domain_quota_total, "sum=", sum(domain_quota_total.values()))
        else:
            print("Synthetic domain quota:", domain_quota, "sum=", total_to_make)
    print("Synthetic dataset root:", synth_root)
    print("Logging to:", log_path)
    print("Batch size:", args.batch_size)
    if args.num_shards > 1:
        print(f"Shard: {args.shard_id}/{args.num_shards}")

    if total_to_make <= 0:
        print("Nothing to generate (synth_total <= 0).")
        return

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.scripts import sample_pair as sample_pair_module

    sample_argv = [
        "--layout_ckpt",
        str(args.layout_ckpt),
        "--layout_checkpoint",
        str(args.layout_checkpoint),
        "--layout_diffusion_type",
        str(args.layout_diffusion_type),
        "--domain",
        "rural",
        "--domain_cond_scale",
        str(args.domain_cond_scale),
        "--controlnet_ckpt",
        str(args.controlnet_ckpt),
        "--save_dir",
        str(tmp_root),
        "--ratios",
        "background:0.2",
        "--prompt",
        args.prompt_rural,
        "--image_size",
        str(args.image_size),
        "--num_inference_steps_layout",
        str(args.num_steps_layout),
        "--num_inference_steps_image",
        str(args.num_steps_image),
        "--batch_size",
        str(args.batch_size),
        "--guidance_scale",
        str(args.guidance_scale),
        "--guidance_rescale",
        str(args.guidance_rescale),
        "--control_scale",
        str(args.control_scale),
        "--lora_scale",
        str(args.lora_scale),
        "--sampler",
        str(args.sampler),
        "--seed",
        str(args.seed),
        "--dtype",
        str(args.dtype),
        "--device",
        str(args.device),
    ]
    if args.base_model:
        sample_argv.extend(["--base_model", str(args.base_model)])

    sample_args = sample_pair_module.parse_args(sample_argv)
    sample_ctx = sample_pair_module.build_context(sample_args)

    with open(log_path, "a", encoding="utf-8") as flog:
        global_i = 0
        while (accepted["rural"] + accepted["urban"]) < total_to_make:
            domain = choose_domain()

            rem = domain_quota[domain] - accepted[domain]
            plan_total = float(cur[domain].sum() + rem * avg_valid_px[domain])
            t = compute_target(domain, args, cached_custom_targets)
            ratios_str, req_dict = build_ratios_str(domain, cur[domain], plan_total, t, args)
            non_bg = {k: v for k, v in req_dict.items() if k != 0}
            if non_bg:
                focus_name = CLASS_NAMES[int(max(non_bg, key=non_bg.get))]
            else:
                focus_name = CLASS_NAMES[0]

            accepted_in_attempt = 0
            for attempt in range(int(args.max_attempts)):
                seed = base_seed + (accepted["rural"] + accepted["urban"]) * 1000 + attempt
                idx = start_idx[domain] + args.shard_id + accepted[domain] * args.num_shards
                tmp_out = tmp_root / f"{domain}_{idx:06d}_try{attempt:02d}"
                if tmp_out.exists():
                    shutil.rmtree(tmp_out, ignore_errors=True)
                tmp_out.mkdir(parents=True, exist_ok=True)

                try:
                    prompt = args.prompt_rural if domain == "rural" else args.prompt_urban
                    run_sample_pair(sample_ctx, sample_pair_module, sample_args, tmp_out, domain, ratios_str, seed, prompt)
                except Exception as exc:
                    record = {
                        "domain": domain,
                        "focus_class": focus_name,
                        "ratios": ratios_str,
                        "seed": seed,
                        "attempt": attempt,
                        "ok": False,
                        "error": f"sample_pair_failed:{type(exc).__name__}",
                        "error_detail": str(exc),
                    }
                    flog.write(json.dumps(record) + "\n")
                    flog.flush()
                    shutil.rmtree(tmp_out, ignore_errors=True)
                    continue

                outputs = load_batch_outputs(tmp_out)
                attempt_ok = False
                for sample in outputs:
                    if accepted[domain] >= domain_quota[domain]:
                        break
                    layout_path = Path(sample["layout_path"])
                    image_path = Path(sample["image_path"])
                    if not layout_path.exists() or not image_path.exists():
                        record = {
                            "domain": domain,
                            "focus_class": focus_name,
                            "ratios": ratios_str,
                            "seed": seed,
                            "attempt": attempt,
                            "sample_index": sample.get("sample_index"),
                            "ok": False,
                            "error": "missing_output_files",
                        }
                        flog.write(json.dumps(record) + "\n")
                        flog.flush()
                        continue

                    m_idx = np.array(Image.open(layout_path))
                    if m_idx.ndim == 3:
                        m_idx = m_idx[:, :, 0]
                    m_idx = m_idx.astype(np.int64)

                    counts = hist_counts(m_idx)
                    p = counts / max(EPS, counts.sum())
                    ok, metrics = accept_layout(
                        domain,
                        p,
                        req_dict,
                        mu_real,
                        t,
                        args,
                    )

                    record = {
                        "domain": domain,
                        "focus_class": focus_name,
                        "ratios": ratios_str,
                        "seed": seed,
                        "attempt": attempt,
                        "sample_index": sample.get("sample_index"),
                        "ok": bool(ok),
                        "metrics": metrics,
                        "p": [float(x) for x in p.tolist()],
                    }
                    flog.write(json.dumps(record) + "\n")
                    flog.flush()

                    if not ok:
                        continue

                    idx = start_idx[domain] + args.shard_id + accepted[domain] * args.num_shards
                    stem = f"{domain}_{idx:06d}"
                    img_dst, msk_dst, rgb_dst, meta_dst, m_idx2 = write_synth_sample(
                        domain,
                        stem,
                        image_path,
                        layout_path,
                        sample.get("layout_color_path"),
                        synth_root,
                        sample.get("metadata"),
                    )
                    cur[domain] += hist_counts(m_idx2)

                    accepted[domain] += 1
                    accepted_in_attempt += 1
                    global_i += 1
                    attempt_ok = True

                    manifest.append(
                        {
                            "domain": domain,
                            "image": str(img_dst),
                            "mask": str(msk_dst),
                            "mask_rgb": str(rgb_dst),
                            "meta": str(meta_dst),
                            "ratios": ratios_str,
                            "focus_class": focus_name,
                            "metrics": metrics,
                        }
                    )

                    if global_i % 25 == 0:
                        print(
                            f"[{global_i}/{total_to_make}] accepted | quotas={domain_quota} | accepted={accepted} "
                            f"| last={domain}/{ratios_str} metrics={metrics}"
                        )

                shutil.rmtree(tmp_out, ignore_errors=True)

                if attempt_ok:
                    break

            if accepted_in_attempt == 0:
                print(f"Warning: too many rejects for {domain} with ratios {ratios_str}.")

    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path
        else synth_root / "meta" / (f"manifest_{args.run_id}.json" if args.run_id else "manifest.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "real_root": str(real_root),
                "synth_root": str(synth_root),
                "synth_total": total_to_make,
                "domain_quota": domain_quota,
                "accepted": accepted,
                "beta": args.beta,
                "gamma": args.gamma,
                "batch_size": args.batch_size,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "items": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Done.")
    print("Synthetic dataset:", synth_root)
    print("Manifest:", manifest_path)
    print("Accepted:", accepted)


if __name__ == "__main__":
    main()
