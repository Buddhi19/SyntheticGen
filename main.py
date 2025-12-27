import argparse
import os
import subprocess
import sys
from pathlib import Path

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "SyntheticGen"
DEFAULT_DATA_ROOT = "/data/inr/llm/Datasets/LOVEDA"
DEFAULT_OUTPUT_DIR = os.path.join(main_dir, "outputs", "checkpoints")
DEFAULT_SAVE_DIR = os.path.join(main_dir, "outputs", "synthetic_predictions")


class ARG:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="SyntheticGen runner for training and generation.")
        self.add_arguments()

    def add_arguments(self) -> None:
        self.parser.add_argument(
            "command",
            choices=["train", "synth"],
            help="Which script to run.",
        )
        self.parser.add_argument(
            "--script_dir",
            type=str,
            default=str(DEFAULT_SCRIPT_DIR),
            help="Path to the diffusers/examples/SyntheticGen scripts.",
        )
        self.parser.add_argument(
            "--data_root",
            type=str,
            default=DEFAULT_DATA_ROOT,
            help="Dataset root used for training (train only).",
        )
        self.parser.add_argument(
            "--dataset",
            type=str,
            default="loveda",
            choices=["loveda", "generic"],
            help="Dataset type for training/synthesis.",
        )
        self.parser.add_argument(
            "--output_dir",
            type=str,
            default=DEFAULT_OUTPUT_DIR,
            help="Output directory for checkpoints (train only).",
        )
        self.parser.add_argument(
            "--checkpoint",
            type=str,
            default=DEFAULT_OUTPUT_DIR,
            help="Checkpoint directory to load (synth only).",
        )
        self.parser.add_argument(
            "--checkpoint_step",
            type=int,
            default=None,
            help="If set, load UNet from checkpoint-<step> under --checkpoint (synth only).",
        )
        self.parser.add_argument(
            "--save_dir",
            type=str,
            default=DEFAULT_SAVE_DIR,
            help="Output directory for generated samples (synth only).",
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
        args.extra_args = extra_args
        return args


def resolve_script(script_dir: str, command: str) -> Path:
    if command == "train":
        script_name = "train_seg_mask_diffusion.py"
    else:
        script_name = "generate_joint_image_mask_diffusion.py"
    script_path = Path(script_dir) / script_name
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Could not find {script_name} at {script_path}. Use --script_dir to set the path."
        )
    return script_path


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
    script_path = resolve_script(args.script_dir, args.command)
    command = [sys.executable, str(script_path), *args.extra_args]
    if args.command == "train":
        _append_if_missing(command, args.extra_args, "--data_root", args.data_root)
        _append_if_missing(command, args.extra_args, "--dataset", args.dataset)
        _append_if_missing(command, args.extra_args, "--output_dir", args.output_dir)
    else:
        _append_if_missing(command, args.extra_args, "--checkpoint", args.checkpoint)
        _append_if_missing(command, args.extra_args, "--checkpoint_step", args.checkpoint_step)
        _append_if_missing(command, args.extra_args, "--save_dir", args.save_dir)
        _append_if_missing(command, args.extra_args, "--dataset", args.dataset)

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


def infer(
    checkpoint: str = DEFAULT_OUTPUT_DIR,
    checkpoint_step: int | None = None,
    save_dir: str = DEFAULT_SAVE_DIR,
    num_samples: int = 5,
    dataset: str = "loveda",
    gpus: str = "6,7",
) -> None:
    script_path = Path(DEFAULT_SCRIPT_DIR) / "generate_joint_image_mask_diffusion.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--checkpoint",
        checkpoint,
    ]
    if checkpoint_step is not None:
        cmd.extend(["--checkpoint_step", str(checkpoint_step)])
    cmd.extend(
        [
            "--save_dir",
            save_dir,
            "--num_samples",
            str(num_samples),
            "--dataset",
            dataset,
        ]
    )
    env = os.environ.copy()
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus.replace(" ", "")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    infer()
