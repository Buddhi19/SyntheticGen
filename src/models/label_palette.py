from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

_LOVEDA_COLORS = {
    "background": (255, 255, 255),
    "building": (255, 0, 0),
    "road": (255, 255, 0),
    "water": (0, 0, 255),
    "barren": (159, 129, 183),
    "forest": (0, 255, 0),
    "agriculture": (255, 195, 128),
    "agricultural": (255, 195, 128),
}

_LOVEDA_CANONICAL = ["background", "building", "road", "water", "barren", "forest", "agriculture"]


def _normalize_class_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def build_palette(
    num_classes: int,
    *,
    dataset: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Returns an (K,3) uint8 palette.

    For LoveDA, uses the canonical color scheme:
      Background=(255,255,255), Building=(255,0,0), Road=(255,255,0), Water=(0,0,255),
      Barren=(159,129,183), Forest=(0,255,0), Agricultural=(255,195,128)
    """
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    if num_classes > 0:
        palette[0] = np.array([0, 0, 0], dtype=np.uint8)

    dataset_norm = str(dataset).strip().lower() if dataset else None
    normalized_names = [_normalize_class_name(x) for x in (class_names or [])]

    is_loveda = dataset_norm == "loveda"
    if not is_loveda and normalized_names:
        canonical_set = {_normalize_class_name(x) for x in _LOVEDA_CANONICAL}
        is_loveda = canonical_set.issubset(set(normalized_names))

    if not is_loveda:
        return palette

    if normalized_names:
        for idx, norm in enumerate(normalized_names[:num_classes]):
            if norm in _LOVEDA_COLORS:
                palette[idx] = np.array(_LOVEDA_COLORS[norm], dtype=np.uint8)
        return palette

    for idx, key in enumerate(_LOVEDA_CANONICAL[:num_classes]):
        palette[idx] = np.array(_LOVEDA_COLORS[key], dtype=np.uint8)
    return palette

