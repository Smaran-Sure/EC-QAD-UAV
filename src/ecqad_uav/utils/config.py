from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_yaml(name: str) -> dict:
    path = ROOT / "configs" / name
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    return ROOT
