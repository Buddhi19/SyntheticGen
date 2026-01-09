from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")

    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    data: Any = None

    if suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml

            data = yaml.safe_load(text)
        except Exception:
            data = json.loads(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping (dict).")

    if isinstance(data.get("args"), dict):
        data = data["args"]
    return data


def apply_config(parser, config_path: str) -> Dict[str, Any]:
    cfg = load_config(config_path)
    cfg = {k: v for k, v in cfg.items() if k != "config"}
    known = {action.dest for action in getattr(parser, "_actions", [])}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {unknown}")
    parser.set_defaults(**cfg)
    return cfg
