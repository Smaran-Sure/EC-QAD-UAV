from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import pandas as pd


class RunLogger:
    def __init__(self):
        self.rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe().to_csv(path, index=False)
        return path


def save_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path
