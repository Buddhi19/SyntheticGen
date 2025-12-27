import argparse
import os
import subprocess
import sys
from pathlib import Path

main_dir = Path(__file__).resolve().parents[2]

DEFAULT_DATA_ROOT = "/data/inr/llm/Datasets/LOVEDA"
DEFAULT_LAYOUT_DIR = os.path.join(str(main_dir), "outputs", "layout_ddpm")
DEFAULT_CONTROLNET_DIR = os.path.join(str(main_dir), "outputs", "controlnet_ratio")
DEFAULT_SAVE_DIR = os.path.join(str(main_dir), "outputs", "synthetic_pairs")


class ARG:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="SyntheticGen runner for training and generation.")
        self.add_arguments()

    def add_arguments(self) -> None:
        self.parser.add_argument(
            "command",
            choices=["train_layout", "train_controlnet", "sample"],
            help="Which script to run.",
        )
        self.parser.add_argument(
            "--data_root",
            type=str,
            default=DEFAULT_DATA_ROOT,
            help="Dataset root used for training.",
        )
        self.parser.add_argument(
            "--dataset",
            type=str,
            default="loveda",
            choices=["loveda", "generic"],
            help="Dataset type for training.",
        )
        self.parser.add_argument(
            "--layout_output_dir",
            type=str,
            default=DEFAULT_LAYOUT_DIR,
            help="Output directory for layout DDPM checkpoints.",
        )
        self.parser.add_argument(
            "--controlnet_output_dir",
            type=str,
            default=DEFAULT_CONTROLNET_DIR,
            help="Output directory for ControlNet checkpoints.",
        )
        self.parser.add_argument(
            "--layout_ckpt",
            type=str,
            default=DEFAULT_LAYOUT_DIR,
            help="Layout DDPM checkpoint directory.",
        )
        self.parser.add_argument(
            "--controlnet_ckpt",
            type=str,
            default=DEFAULT_CONTROLNET_DIR,
            help="ControlNet checkpoint directory.",
        )
        self.parser.add_argument(
            "--save_dir",
            type=str,
            default=DEFAULT_SAVE_DIR,
            help="Output directory for generated samples.",
        )
        self.parser.add_argument(
            "--base_model",
            type=str,
            default=None,
            help="Base Stable Diffusion model path for ControlNet training/sampling.",
        )
        self.parser.add_argument(
            "--gpus",
            type=str,
            default="5,6,7",
            help="Comma-separated GPU ids to expose via CUDA_VISIBLE_DEVICES.",
        )
        self.parser.add_argument(
            "--dry_run",
            action="store_true",
            help="Print the resolved command without executing it.",
        )

    def parse(self) -> argparse.Namespace:
        args, extra_args = self.parser.parse_known_args()
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]
        extra_args, args = _consume_forwarded_args(extra_args, args)
        args.extra_args = extra_args
        return args


def _consume_forwarded_args(extra_args, args: argparse.Namespace):
    def consume(flag: str):
        value = None
        remainder = []
        idx = 0
        while idx < len(extra_args):
            arg = extra_args[idx]
            if arg == flag:
                if idx + 1 >= len(extra_args):
                    raise ValueError(f"{flag} requires a value")
                value = extra_args[idx + 1]
                idx += 2
                continue
            if arg.startswith(f"{flag}="):
                value = arg.split("=", 1)[1]
                idx += 1
                continue
            remainder.append(arg)
            idx += 1
        return value, remainder

    for flag, attr in (
        ("--data_root", "data_root"),
        ("--dataset", "dataset"),
        ("--layout_output_dir", "layout_output_dir"),
        ("--controlnet_output_dir", "controlnet_output_dir"),
        ("--layout_ckpt", "layout_ckpt"),
        ("--controlnet_ckpt", "controlnet_ckpt"),
        ("--save_dir", "save_dir"),
        ("--base_model", "base_model"),
        ("--gpus", "gpus"),
    ):
        value, extra_args = consume(flag)
        if value is not None:
            setattr(args, attr, value)
    return extra_args, args


def _flag_present(extra_args, flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in extra_args)


def _append_if_missing(command, extra_args, flag: str, value: str) -> None:
    if value is None:
        return
    if _flag_present(extra_args, flag):
        return
    command.extend([flag, str(value)])


def main() -> None:
    args = ARG().parse()
    scripts = {
        "train_layout": "train_layout_ddpm.py",
        "train_controlnet": "train_controlnet_ratio.py",
        "sample": "sample_pair.py",
    }
    script_path = Path(__file__).resolve().parent / scripts[args.command]
    command = [sys.executable, str(script_path), *args.extra_args]
    if args.command == "train_layout":
        _append_if_missing(command, args.extra_args, "--data_root", args.data_root)
        _append_if_missing(command, args.extra_args, "--dataset", args.dataset)
        _append_if_missing(command, args.extra_args, "--output_dir", args.layout_output_dir)
    elif args.command == "train_controlnet":
        _append_if_missing(command, args.extra_args, "--data_root", args.data_root)
        _append_if_missing(command, args.extra_args, "--dataset", args.dataset)
        _append_if_missing(command, args.extra_args, "--output_dir", args.controlnet_output_dir)
        _append_if_missing(command, args.extra_args, "--pretrained_model_name_or_path", args.base_model)
    else:
        _append_if_missing(command, args.extra_args, "--layout_ckpt", args.layout_ckpt)
        _append_if_missing(command, args.extra_args, "--controlnet_ckpt", args.controlnet_ckpt)
        _append_if_missing(command, args.extra_args, "--save_dir", args.save_dir)
        _append_if_missing(command, args.extra_args, "--base_model", args.base_model)

    gpus = None
    if args.gpus is not None:
        gpus = args.gpus.replace(" ", "")
        if not gpus:
            raise ValueError("--gpus cannot be empty.")
    if args.dry_run:
        prefix = f"CUDA_VISIBLE_DEVICES={gpus} " if gpus else ""
        print(prefix + " ".join(command))
        return

    env = os.environ.copy()
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
