from collections import OrderedDict
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

COLOR_MAP_LOVEDA = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)

LABEL_MAP_LOVEDA = OrderedDict(
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6,
)


def _normalize_class_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


_LOVEDA_CANONICAL_BY_NORMALIZED = {
    "background": "Background",
    "building": "Building",
    "road": "Road",
    "water": "Water",
    "barren": "Barren",
    "forest": "Forest",
    "agriculture": "Agricultural",
    "agricultural": "Agricultural",
}


def build_palette(class_names: Sequence[str], num_classes: int, dataset: Optional[str] = None) -> np.ndarray:
    names = [str(x) for x in class_names[:num_classes]]
    looks_like_loveda = all(_normalize_class_name(name) in _LOVEDA_CANONICAL_BY_NORMALIZED for name in names)
    if dataset == "loveda" or looks_like_loveda:
        palette = np.zeros((num_classes, 3), dtype=np.uint8)
        for idx, name in enumerate(names):
            canonical = _LOVEDA_CANONICAL_BY_NORMALIZED.get(_normalize_class_name(name))
            if canonical is None:
                raise ValueError(f"Unknown LoveDA class name: {name}")
            palette[idx] = np.array(COLOR_MAP_LOVEDA[canonical], dtype=np.uint8)
        return palette

    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)
    return colors


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
        data = json.loads(path.read_text())
        if isinstance(data, list):
            class_names = [str(x) for x in data]
        elif isinstance(data, dict):
            keys = sorted(int(k) for k in data.keys())
            class_names = [None] * (keys[-1] + 1)
            for k in keys:
                class_names[k] = str(data[str(k)] if str(k) in data else data[k])
            if any(x is None for x in class_names):
                raise ValueError("class_names_json keys must be contiguous from 0")
        else:
            raise ValueError("class_names_json must be list or dict")
        if num_classes is None:
            num_classes = len(class_names)
        if num_classes != len(class_names):
            raise ValueError("num_classes != len(class_names)")
        return class_names, num_classes

    if dataset == "loveda":
        class_names = DEFAULT_LOVEDA_CLASS_NAMES
        if num_classes is None:
            num_classes = len(class_names)
        if num_classes != len(class_names):
            raise ValueError("num_classes != LoveDA classes")
        return class_names, num_classes

    if num_classes is None:
        raise ValueError("num_classes must be provided when class_names_json is not set")
    return [f"class_{i}" for i in range(num_classes)], num_classes


def _list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS])


def _pair_by_stem(image_paths: Sequence[Path], mask_paths: Sequence[Path]) -> List[Tuple[Path, Path]]:
    mask_by_stem = {p.stem: p for p in mask_paths}
    pairs = []
    for ip in image_paths:
        mp = mask_by_stem.get(ip.stem)
        if mp is not None:
            pairs.append((ip, mp))
    return sorted(pairs, key=lambda x: x[0].name)


def _find_dir(root: Path, candidates: Iterable[str]) -> Optional[Path]:
    for c in candidates:
        p = root / c
        if p.is_dir():
            return p
    return None


def _discover_pairs(root: Path) -> List[Tuple[Path, Path]]:
    image_candidates = ["images_png", "images", "imgs", "image", "JPEGImages"]
    mask_candidates = ["masks_png", "masks", "labels", "label", "annotations", "SegmentationClass"]
    image_dir = _find_dir(root, image_candidates)
    mask_dir = _find_dir(root, mask_candidates)
    if image_dir is None or mask_dir is None:
        raise FileNotFoundError(f"Could not find image/mask dirs under {root}")
    image_paths = _list_images(image_dir)
    mask_paths = _list_images(mask_dir)
    pairs = _pair_by_stem(image_paths, mask_paths)
    if not pairs:
        raise FileNotFoundError("No matching image/mask pairs found (stem mismatch).")
    return pairs


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


def _load_image(path: Path, image_size: int) -> torch.FloatTensor:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_label(path: Path, image_size: int, label_remap: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> torch.LongTensor:
    label = Image.open(path)
    label = ImageOps.exif_transpose(label)
    if label.mode not in {"L", "P"}:
        label = label.convert("L")
    if image_size is not None:
        label = label.resize((image_size, image_size), resample=Image.NEAREST)
    arr = np.asarray(label, dtype=np.int64)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if label_remap is not None:
        arr = label_remap(arr)
    return torch.from_numpy(arr)


def _label_to_onehot_and_ratios(label: torch.Tensor, num_classes: int, ignore_index: Optional[int]):
    # label: (H,W) long
    if ignore_index is None:
        valid = torch.ones_like(label, dtype=torch.bool)
        safe = label
    else:
        valid = label != ignore_index
        safe = label.clone()
        safe[~valid] = 0

    onehot = F.one_hot(safe, num_classes=num_classes).permute(2, 0, 1).float()
    onehot = onehot * valid.unsqueeze(0).float()

    counts = onehot.sum(dim=(1, 2))
    denom = counts.sum().clamp(min=1.0)
    ratios = counts / denom
    return onehot, ratios, valid.unsqueeze(0).float()


class _SegmentationDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        image_size: int,
        num_classes: int,
        ignore_index: Optional[int],
        return_layouts: bool,
        layout_size: int = 64,
        label_remap: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.pairs = list(pairs)
        self.image_size = image_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.return_layouts = return_layouts
        self.layout_size = layout_size
        self.label_remap = label_remap

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]
        image = _load_image(image_path, self.image_size)
        label = _load_label(mask_path, self.image_size, self.label_remap)

        out = {
            "image": image,
            "pixel_values": image,
            "label": label,
            "labels": label,
            "image_path": str(image_path),
            "label_path": str(mask_path),
        }

        if self.return_layouts:
            onehot_512, ratios, valid_512 = _label_to_onehot_and_ratios(label, self.num_classes, self.ignore_index)
            onehot_small = F.interpolate(
                onehot_512.unsqueeze(0),
                size=(self.layout_size, self.layout_size),
                mode="nearest",
            ).squeeze(0)
            valid_small = F.interpolate(
                valid_512.unsqueeze(0),
                size=(self.layout_size, self.layout_size),
                mode="nearest",
            ).squeeze(0)

            out.update(
                {
                    "layout_512": onehot_512,
                    "valid_512": valid_512,
                    # Backward compatible keys (Stage B / older code)
                    "layout_64": onehot_small,
                    "valid_64": valid_small,
                    # Explicit keys (Stage A uses these)
                    f"layout_{self.layout_size}": onehot_small,
                    f"valid_{self.layout_size}": valid_small,
                    "ratios": ratios,
                }
            )
        if hasattr(self, "sample_domains"):
            out["domain"] = self.sample_domains[idx]
        return out


class GenericSegDataset(_SegmentationDataset):
    def __init__(
        self,
        root: str,
        image_size: int,
        num_classes: int,
        ignore_index: Optional[int] = None,
        return_layouts: bool = False,
        layout_size: int = 64,
    ):
        root_path = Path(root)
        pairs = _discover_pairs(root_path)
        super().__init__(
            pairs=pairs,
            image_size=image_size,
            num_classes=num_classes,
            ignore_index=ignore_index,
            return_layouts=return_layouts,
            layout_size=layout_size,
            label_remap=None,
        )


class LoveDADataset(_SegmentationDataset):
    def __init__(
        self,
        root: str,
        image_size: int,
        split: str = "Train",
        domains: Sequence[str] = ("Urban", "Rural"),
        ignore_index: int = 255,
        num_classes: int = len(DEFAULT_LOVEDA_CLASS_NAMES),
        return_layouts: bool = False,
        layout_size: int = 64,
    ):
        root_path = Path(root)
        split_dir = _resolve_split_dir(root_path, split)
        if split_dir is None:
            raise FileNotFoundError(f"Could not find split '{split}' under {root}.")
        pairs: List[Tuple[Path, Path]] = []
        sample_domains: List[str] = []

        for domain in domains:
            domain_dir = _resolve_domain_dir(split_dir, domain)
            if domain_dir is None:
                continue
            try:
                dom_pairs = _discover_pairs(domain_dir)
            except FileNotFoundError:
                continue

            pairs.extend(dom_pairs)
            sample_domains.extend([str(domain).lower()] * len(dom_pairs))
        if not pairs:
            raise FileNotFoundError("No LoveDA pairs found. Check split/domains structure.")

        label_remap = lambda arr, ii=ignore_index: remap_loveda_labels(arr, ii)
        super().__init__(
            pairs=pairs,
            image_size=image_size,
            num_classes=num_classes,
            ignore_index=ignore_index,
            return_layouts=return_layouts,
            layout_size=layout_size,
            label_remap=label_remap,
        )
        self.sample_domains = sample_domains
