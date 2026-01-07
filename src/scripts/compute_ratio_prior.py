#!/usr/bin/env python
# coding=utf-8

"""Compute a global class-ratio prior from a segmentation dataset."""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .dataset_loveda import GenericSegDataset, LoveDADataset, load_class_names
except ImportError:  # direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.scripts.dataset_loveda import GenericSegDataset, LoveDADataset, load_class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Compute a global class-ratio prior (mean ratios over dataset).")
    parser.add_argument("--data_root", type=str, required=True, help="Root folder for the dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="generic",
        choices=["loveda", "generic"],
        help="Dataset type to use.",
    )
    parser.add_argument("--class_names_json", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--loveda_split", type=str, default="Train")
    parser.add_argument("--loveda_domains", type=str, default="Urban,Rural")
    parser.add_argument("--image_size", type=int, default=512, help="Resize masks before ratio computation.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_path", type=str, default="outputsV2/ratio_prior.json")
    return parser.parse_args()


def _resolve_dataset(args, num_classes: int):
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
            layout_size=64,
        )
    return GenericSegDataset(
        args.data_root,
        image_size=args.image_size,
        num_classes=num_classes,
        ignore_index=args.ignore_index,
        return_layouts=True,
        layout_size=64,
    )


def main():
    args = parse_args()
    class_names, num_classes = load_class_names(args.class_names_json, args.num_classes, args.dataset)
    dataset = _resolve_dataset(args, num_classes)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    ratio_sum = torch.zeros((num_classes,), dtype=torch.float64)
    count = 0
    for batch in loader:
        ratios = batch["ratios"]  # (B,K)
        ratio_sum += ratios.double().sum(dim=0)
        count += int(ratios.shape[0])

    if count <= 0:
        raise ValueError("No samples found to compute ratio prior.")
    ratio_prior = (ratio_sum / float(count)).clamp(min=0)
    ratio_prior = ratio_prior / ratio_prior.sum().clamp(min=1e-12)

    payload = {
        "dataset": args.dataset,
        "data_root": str(args.data_root),
        "image_size": int(args.image_size),
        "num_samples": int(count),
        "class_names": class_names,
        "ratio_prior": [float(x) for x in ratio_prior.tolist()],
    }

    out_path = Path(args.output_path)
    os.makedirs(out_path.parent, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote ratio prior to {out_path}")


if __name__ == "__main__":
    main()

