import json
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_LOVEDA_CLASS_NAMES = [
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
]


def remap_loveda_labels(arr: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    # LoveDA: 0 = no-data (ignore), 1..7 = classes
    arr = arr.astype(np.int64)
    ignore = arr == 0
    arr[ignore] = ignore_index
    arr[~ignore] = arr[~ignore] - 1
    return arr


def load_class_names(
    class_names_json: Optional[str],
    num_classes: Optional[int] = None,
    dataset: Optional[str] = None,
) -> Tuple[List[str], int]:
    if class_names_json is not None:
        path = Path(class_names_json)
        if not path.exists():
            raise FileNotFoundError(f"class_names_json not found: {class_names_json}")
        data = json.loads(path.read_text())
        if isinstance(data, list):
            class_names = [str(name) for name in data]
        elif isinstance(data, dict):
            try:
                keys = sorted(int(k) for k in data.keys())
            except ValueError as exc:
                raise ValueError("class_names_json dict keys must be integers") from exc
            max_key = keys[-1]
            class_names = [None] * (max_key + 1)
            for key in keys:
                value = data[str(key)] if str(key) in data else data[key]
                class_names[key] = str(value)
            if any(name is None for name in class_names):
                raise ValueError("class_names_json dict must contain contiguous keys starting at 0")
        else:
            raise ValueError("class_names_json must be a list or dict")
        if num_classes is None:
            num_classes = len(class_names)
        elif num_classes != len(class_names):
            raise ValueError(
                f"num_classes ({num_classes}) does not match class_names_json length ({len(class_names)})"
            )
        return class_names, num_classes

    if dataset == "loveda":
        class_names = DEFAULT_LOVEDA_CLASS_NAMES
        if num_classes is None:
            num_classes = len(class_names)
        elif num_classes != len(class_names):
            raise ValueError(
                f"num_classes ({num_classes}) does not match LoveDA classes ({len(class_names)})"
            )
        return class_names, num_classes

    if num_classes is None:
        raise ValueError("num_classes must be provided when class_names_json is not set")
    return [f"class_{i}" for i in range(num_classes)], num_classes


def _list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS])


def _pair_by_stem(image_paths: Sequence[Path], mask_paths: Sequence[Path]) -> List[Tuple[Path, Path]]:
    mask_by_stem = {path.stem: path for path in mask_paths}
    pairs = []
    for image_path in image_paths:
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return sorted(pairs, key=lambda pair: pair[0].name)


def _find_dir(root: Path, candidates: Iterable[str]) -> Optional[Path]:
    for candidate in candidates:
        candidate_path = root / candidate
        if candidate_path.is_dir():
            return candidate_path
    return None


def _resolve_split_dir(root: Path, split: str) -> Optional[Path]:
    direct = root / split
    if direct.is_dir():
        return direct
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.lower() == split.lower():
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


def _discover_pairs(root: Path) -> List[Tuple[Path, Path]]:
    image_candidates = ["images_png", "images", "imgs", "image", "JPEGImages"]
    mask_candidates = ["masks_png", "masks", "labels", "label", "annotations", "SegmentationClass"]

    image_dir = _find_dir(root, image_candidates)
    mask_dir = _find_dir(root, mask_candidates)
    if image_dir is None or mask_dir is None:
        raise FileNotFoundError(
            "Could not find image/mask directories in "
            f"{root}. Looked for images in {image_candidates} and masks in {mask_candidates}."
        )

    image_paths = _list_images(image_dir)
    mask_paths = _list_images(mask_dir)
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")
    if not mask_paths:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")

    pairs = _pair_by_stem(image_paths, mask_paths)
    if not pairs:
        raise FileNotFoundError(
            "No matching image/mask pairs found. Make sure filenames match between "
            f"{image_dir} and {mask_dir}."
        )
    return pairs


def _load_image(path: Path, image_size: int) -> torch.FloatTensor:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    array = array / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def _load_label(
    path: Path, image_size: int, label_remap: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> torch.LongTensor:
    label = Image.open(path)
    label = ImageOps.exif_transpose(label)
    if label.mode not in {"L", "P"}:
        label = label.convert("L")
    if image_size is not None:
        label = label.resize((image_size, image_size), resample=Image.NEAREST)
    array = np.asarray(label, dtype=np.int64)
    if array.ndim == 3:
        array = array[:, :, 0]
    if label_remap is not None:
        array = label_remap(array)
    return torch.from_numpy(array)


class _SegmentationDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        image_size: int,
        label_remap: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        num_classes: Optional[int] = None,
        ignore_index: Optional[int] = None,
        return_layouts: bool = False,
    ):
        if not pairs:
            raise ValueError("No image/mask pairs found.")
        self.pairs = list(pairs)
        self.image_size = image_size
        self.label_remap = label_remap
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.return_layouts = return_layouts

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = _load_image(image_path, self.image_size)
        label = _load_label(mask_path, self.image_size, self.label_remap)
        sample = {
            "image": image,
            "label": label,
            "pixel_values": image,
            "labels": label,
            "image_path": str(image_path),
            "label_path": str(mask_path),
        }
        if self.return_layouts and self.num_classes is not None:
            valid = torch.ones_like(label, dtype=torch.bool)
            if self.ignore_index is not None:
                valid = label != self.ignore_index
            label_clamped = label.clone()
            label_clamped[~valid] = 0
            onehot = F.one_hot(label_clamped.long(), num_classes=self.num_classes).permute(2, 0, 1).float()
            if self.ignore_index is not None:
                onehot = onehot * valid.float().unsqueeze(0)
            if valid.any():
                flat = label[valid].view(-1)
                counts = torch.bincount(flat.long(), minlength=self.num_classes).float()
            else:
                counts = torch.zeros(self.num_classes, dtype=torch.float32)
            ratios = counts / counts.sum().clamp(min=1.0)
            layout_64 = F.interpolate(onehot.unsqueeze(0), size=(64, 64), mode="nearest").squeeze(0)
            sample.update(
                {
                    "layout_512": onehot,
                    "layout_64": layout_64,
                    "ratios": ratios,
                }
            )
        return sample


class GenericSegDataset(_SegmentationDataset):
    def __init__(
        self,
        root: str,
        image_size: int,
        num_classes: Optional[int] = None,
        ignore_index: Optional[int] = None,
        return_layouts: bool = False,
    ):
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"data_root does not exist: {root}")
        pairs = _discover_pairs(root_path)
        super().__init__(
            pairs,
            image_size,
            num_classes=num_classes,
            ignore_index=ignore_index,
            return_layouts=return_layouts,
        )


class LoveDADataset(_SegmentationDataset):
    def __init__(
        self,
        root: str,
        image_size: int,
        split: str = "Train",
        domains: Sequence[str] = ("Urban", "Rural"),
        ignore_index: int = 255,
        num_classes: Optional[int] = None,
        return_layouts: bool = False,
    ):
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"data_root does not exist: {root}")

        split_dir = _resolve_split_dir(root_path, split)
        if split_dir is None:
            raise FileNotFoundError(
                f"Could not find split '{split}' under {root}. Expected directories like {root}/Train or {root}/Val."
            )

        pairs: List[Tuple[Path, Path]] = []
        for domain in domains:
            domain_dir = _resolve_domain_dir(split_dir, domain)
            if domain_dir is None:
                continue
            try:
                pairs.extend(_discover_pairs(domain_dir))
            except FileNotFoundError:
                continue

        if not pairs:
            raise FileNotFoundError(
                "No LoveDA image/mask pairs found. Expected structure like "
                f"{root}/Train/Urban/images_png and {root}/Train/Urban/masks_png."
            )

        label_remap = lambda array, ignore_index=ignore_index: remap_loveda_labels(array, ignore_index)
        super().__init__(
            pairs,
            image_size,
            label_remap=label_remap,
            num_classes=num_classes,
            ignore_index=ignore_index,
            return_layouts=return_layouts,
        )
