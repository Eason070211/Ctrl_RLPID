"""YAML config loader with defaults."""
from __future__ import annotations
import os
import yaml

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(_HERE, "configs", "default.yaml")


def load_config(path: str | None = None) -> dict:
    path = path or DEFAULT_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
