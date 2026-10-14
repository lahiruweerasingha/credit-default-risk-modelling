"""Configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/pipeline.yaml")


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the pipeline config. `CREDITRISK_CONFIG` overrides the default path."""
    cfg_path = Path(path or os.environ.get("CREDITRISK_CONFIG", DEFAULT_CONFIG))
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def data_path(cfg: dict[str, Any], key: str) -> Path:
    return Path(cfg["data"][key])


def models_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg.get("outputs", {}).get("models_dir", "data/models"))


def reports_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg.get("outputs", {}).get("reports_dir", "reports"))
