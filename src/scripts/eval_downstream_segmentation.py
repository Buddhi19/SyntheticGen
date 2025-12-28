#!/usr/bin/env python
# coding=utf-8

"""Train a simple segmentation model on real vs real+synthetic and report mIoU."""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

try:
    from .dataset_loveda import GenericSegDataset, LoveDADataset, load_class_names
    from ..models.segmentation import SimpleSegNet
except ImportError:  # direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.scripts.dataset_loveda import GenericSegDataset, LoveDADataset, load_class_names
    from src.models.segmentation import SimpleSegNet


logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    mean_iou: float
    per_class_iou: list


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate downstream segmentation utility.")
    parser.add_argument("--real_data_root", type=str, required=True)
    parser.add_argument("--synthetic_data_root", type=str, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default="generic",
        choices=["loveda", "generic"],
        help="Dataset type for real data.",
    )
    parser.add_argument("--synthetic_dataset", type=str, default="generic", choices=["generic", "loveda"])
    parser.add_argument("--class_names_json", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_split", type=str, default="Train")
    parser.add_argument("--val_split", type=str, default="Val")
    parser.add_argument("--loveda_domains", type=str, default="Urban,Rural")
    parser.add_argument("--val_data_root", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    default_output_path = str(Path(__file__).resolve().parents[2] / "outputsimproved" / "eval_downstream_segmentation.json")
    parser.add_argument("--output_path", type=str, default=default_output_path)
    parser.add_argument(
        "--save_ckpt",
        type=str,
        default=None,
        help="Optional path to save the trained real-only segmentation model (state_dict).",
    )
    return parser.parse_args()


def _resolve_dataset(
    root: str,
    dataset_type: str,
    image_size: int,
    split: str,
    domains,
    num_classes: int,
    ignore_index: int,
):
    if dataset_type == "loveda":
        return LoveDADataset(
            root,
            image_size=image_size,
            split=split,
            domains=domains,
            ignore_index=ignore_index,
            num_classes=num_classes,
            return_layouts=False,
        )
    return GenericSegDataset(
        root,
        image_size=image_size,
        num_classes=num_classes,
        ignore_index=ignore_index,
        return_layouts=False,
    )


def _update_confusion(conf: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int):
    mask = targets != ignore_index
    preds = preds[mask].view(-1)
    targets = targets[mask].view(-1)
    if preds.numel() == 0:
        return
    idx = targets * num_classes + preds
    conf += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _compute_iou(conf: torch.Tensor) -> Metrics:
    conf = conf.float()
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = (tp + fp + fn).clamp(min=1.0)
    iou = (tp / denom).cpu().tolist()
    mean_iou = float(sum(iou) / len(iou)) if iou else 0.0
    return Metrics(mean_iou=mean_iou, per_class_iou=iou)


def _train_and_eval(train_dataset, val_dataset, args, save_ckpt_path: Optional[str] = None) -> Metrics:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    model = SimpleSegNet(args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=2,
    )

    model.train()
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in train_loader:
            images = batch["pixel_values"].to(device=device, dtype=torch.float32)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, ignore_index=args.ignore_index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step >= args.max_train_steps:
                break

    model.eval()
    conf = torch.zeros((args.num_classes, args.num_classes), device=device)
    with torch.no_grad():
        for batch in val_loader:
            images = batch["pixel_values"].to(device=device, dtype=torch.float32)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            _update_confusion(conf, preds, labels, args.num_classes, args.ignore_index)

    metrics = _compute_iou(conf)
    if save_ckpt_path:
        save_path = Path(save_ckpt_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return metrics


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    class_names, num_classes = load_class_names(args.class_names_json, args.num_classes, args.dataset)
    args.num_classes = num_classes

    domains = [d.strip() for d in args.loveda_domains.split(",") if d.strip()]
    train_real = _resolve_dataset(
        args.real_data_root,
        args.dataset,
        args.image_size,
        args.train_split,
        domains,
        num_classes,
        args.ignore_index,
    )

    if args.dataset == "loveda":
        val_root = args.real_data_root
    else:
        if args.val_data_root is None:
            raise ValueError("--val_data_root is required for generic datasets.")
        val_root = args.val_data_root

    val_dataset = _resolve_dataset(
        val_root,
        args.dataset,
        args.image_size,
        args.val_split,
        domains,
        num_classes,
        args.ignore_index,
    )

    metrics_real = _train_and_eval(train_real, val_dataset, args, save_ckpt_path=args.save_ckpt)

    results = {
        "real_only": {
            "mean_iou": metrics_real.mean_iou,
            "per_class_iou": metrics_real.per_class_iou,
            "class_names": class_names,
        }
    }

    if args.synthetic_data_root:
        train_synth = _resolve_dataset(
            args.synthetic_data_root,
            args.synthetic_dataset,
            args.image_size,
            args.train_split,
            domains,
            num_classes,
            args.ignore_index,
        )
        train_combo = ConcatDataset([train_real, train_synth])
        metrics_combo = _train_and_eval(train_combo, val_dataset, args)
        results["real_plus_synth"] = {
            "mean_iou": metrics_combo.mean_iou,
            "per_class_iou": metrics_combo.per_class_iou,
            "class_names": class_names,
        }

    os.makedirs(Path(args.output_path).parent, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
