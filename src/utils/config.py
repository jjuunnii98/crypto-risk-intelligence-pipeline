from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to config.yaml

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must be a valid YAML mapping (dictionary).")

    return config


def get_target_markets(config: dict[str, Any]) -> list[str]:
    """
    Extract target market list from config.
    """
    markets = config.get("assets", {}).get("target_markets", [])
    if not isinstance(markets, list) or not markets:
        raise ValueError("assets.target_markets must be a non-empty list.")
    return markets


def get_upbit_base_url(config: dict[str, Any]) -> str:
    """
    Extract Upbit API base URL from config.
    """
    base_url = config.get("api", {}).get("upbit", {}).get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("api.upbit.base_url must be a non-empty string.")
    return base_url


def get_raw_data_dir(config: dict[str, Any]) -> Path:
    """
    Extract raw data directory path from config.
    """
    raw_dir = config.get("paths", {}).get("raw_dir")
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        raise ValueError("paths.raw_dir must be a non-empty string.")
    return Path(raw_dir)


def ensure_data_directories(config: dict[str, Any]) -> None:
    """
    Ensure that raw / processed / events directories exist.
    """
    path_keys = ["raw_dir", "processed_dir", "events_dir"]
    paths_cfg = config.get("paths", {})

    for key in path_keys:
        dir_str = paths_cfg.get(key)
        if isinstance(dir_str, str) and dir_str.strip():
            Path(dir_str).mkdir(parents=True, exist_ok=True)