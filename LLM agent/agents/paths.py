"""Resolve paths relative to the LLM agent project root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def resolve_config(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_path(relative: str) -> Path:
    return PROJECT_ROOT / relative
