"""Configuration loader for MRL demo and generative search summarization."""

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("mrl.config")

DEFAULT_CONFIG: dict[str, Any] = {
    "retrieval": {
        "model_path": "checkpoints/best_model",
        "data_dir": "data/processed",
        "default_dimension": 128,
        "top_k": 5,
        "device": "mps",
    },
    "summary": {
        "enabled": True,
        "stream": True,
        "default_backend": "auto",
        "max_new_tokens": 384,
        "temperature": 0.3,
        "top_p": 0.9,
        "local": {
            "model_name": "google/gemma-2-2b-it",
            "torch_dtype": "float16",
            "device": "mps",
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "model_name": "gemma2:2b",
            "timeout": 30.0,
        },
        "google": {
            "model_name": "gemini-2.0-flash",
            "timeout": 20.0,
        },
    },
}


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge source dictionary into target dictionary."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


class ConfigDict(dict):
    """Dictionary subclass that provides attribute-style dot access."""

    def __getattr__(self, name: str) -> Any:
        if name in self:
            val = self[name]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                val = ConfigDict(val)
                self[name] = val
            return val
        raise AttributeError(f"Configuration has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"Configuration has no attribute '{name}'")


def load_config(config_path: Path | str | None = None) -> ConfigDict:
    """Load configuration from YAML file and merge with default schema."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    _deep_merge(config, loaded)
                    logger.info(f"Loaded configuration from {path}")
                else:
                    logger.warning(f"Config file {path} is not a valid YAML dictionary; using defaults.")
            except Exception as e:
                logger.warning(f"Failed to read config file {path}: {e}; using defaults.")
        else:
            logger.warning(f"Specified config file {path} not found; using defaults.")
    else:
        # Check standard default locations
        standard_path = Path("configs/config.yaml")
        if standard_path.exists():
            try:
                with open(standard_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    _deep_merge(config, loaded)
                    logger.info(f"Loaded configuration from default path {standard_path}")
            except Exception as e:
                logger.warning(f"Error loading {standard_path}: {e}")

    return ConfigDict(config)
