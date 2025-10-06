"""Dict-based configuration for uma_geometry_optimizer.

This module provides a simple JSON/YAML-backed configuration represented as a
nested Python dict while still exposing a Config class API. Unknown keys are
preserved, and only a small set of known keys have defaults derived from
examples/config.json so users can add new keys without changing the code.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional

# Default configuration aligned with examples/config.json
DEFAULT_CONFIG: Dict[str, Any] = {
    "optimization": {
        "batch_optimization_mode": "batch",
        "batch_optimizer": "fire",
        "max_num_conformers": 20,
        "conformer_seed": 42,

        "model_name": "uma-s-1p1",
        "model_path": None,
        "device": "cuda",
        "huggingface_token": None,
        "huggingface_token_file": "/home/employee/n_hoel07/hf_secret",

        "verbose": True,
    }
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dicts, with values from override taking precedence.

    Leaves inputs unmodified and returns a new merged dict.
    """
    result = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class _Section:
    """Lightweight wrapper to provide attribute access to a nested dict section."""

    def __init__(self, root: Dict[str, Any], path: list[str]):
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_path", path)

    def _node(self) -> Dict[str, Any]:
        node = self._root
        for key in self._path:
            node = node.setdefault(key, {})
        return node

    def __getattr__(self, name: str):
        node = self._node()
        if name in node:
            val = node[name]
            if isinstance(val, dict):
                return _Section(self._root, self._path + [name])
            return val
        raise AttributeError(f"{name} not found in section {'.'.join(self._path) if self._path else 'root'}")

    def __setattr__(self, name: str, value: Any) -> None:
        node = self._node()
        node[name] = value

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._node())

    def get(self, key: str, default: Any = None) -> Any:
        return self._node().get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        return self._node().setdefault(key, default)

    # Specific helper used by model.py in older API
    def get_huggingface_token(self) -> Optional[str]:
        opt = self._node()
        token = opt.get("huggingface_token")
        if token:
            return str(token)
        token_file = opt.get("huggingface_token_file")
        if not token_file:
            return None
        try:
            with open(str(token_file), "r", encoding="utf-8") as fh:
                content = fh.read().strip()
            return content or None
        except (OSError, IOError) as e:
            if bool(opt.get("verbose", True)):
                print(f"Warning: Could not read HuggingFace token from {token_file}: {e}")
            return None


class Config:
    """Dict-backed configuration with attribute access for sections.

    Example:
        cfg = load_config_from_file()
        print(cfg.optimization.verbose)
        cfg.optimization.device = "cuda"
        save_config_to_file(cfg, "config.json")
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        merged = _deep_merge(DEFAULT_CONFIG, data or {})
        self._data: Dict[str, Any] = merged

    @property
    def optimization(self) -> _Section:
        return _Section(self._data, ["optimization"])

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(data)


# Backward-compat: expose a symbol named OptimizationConfig
class OptimizationConfig(_Section):
    pass


def load_config_from_file(filepath: str = "config.json") -> Config:
    """Load configuration from a JSON/YAML file and deep-merge with defaults.

    Args:
        filepath: Path to the config file. If it doesn't exist, returns defaults.

    Returns:
        Config instance with data merged with DEFAULT_CONFIG. Unknown keys are preserved.
    """
    if not os.path.exists(filepath):
        return Config()

    with open(filepath, "r", encoding="utf-8") as f:
        if filepath.endswith(".json"):
            user_cfg = json.load(f)
        elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
            try:
                from yaml import safe_load as _yaml_safe_load  # type: ignore
                user_cfg = _yaml_safe_load(f)
            except ImportError as e:
                raise ImportError("PyYAML is required to load YAML config files") from e
        else:
            raise ValueError("Config file must be JSON or YAML format")

    if not isinstance(user_cfg, dict):
        raise ValueError("Configuration file must contain a JSON/YAML object at the root")

    return Config.from_dict(user_cfg)


def save_config_to_file(config: Any, filepath: str) -> None:
    """Save configuration to JSON/YAML file.

    Accepts either a Config instance or a plain dict.
    """
    cfg_dict = config.to_dict() if isinstance(config, Config) else dict(config)

    with open(filepath, "w", encoding="utf-8") as f:
        if filepath.endswith(".json"):
            json.dump(cfg_dict, f, indent=2)
        elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
            try:
                from yaml import safe_dump as _yaml_safe_dump  # type: ignore
                _yaml_safe_dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
            except ImportError as e:
                raise ImportError("PyYAML is required to save YAML config files") from e
        else:
            raise ValueError("Config file must be JSON or YAML format")


def get_huggingface_token(config: Config | Dict[str, Any]) -> Optional[str]:
    """Return the HuggingFace token from config or a file, if available.

    Works with either Config or a plain dict. Checks optimization.huggingface_token first,
    then optimization.huggingface_token_file.
    """
    if isinstance(config, Config):
        return config.optimization.get_huggingface_token()

    opt = (config or {}).get("optimization", {})
    token = opt.get("huggingface_token")
    if token:
        return str(token)

    token_file = opt.get("huggingface_token_file")
    if not token_file:
        return None

    try:
        with open(str(token_file), "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        return content or None
    except (OSError, IOError) as e:
        if bool(opt.get("verbose", True)):
            print(f"Warning: Could not read HuggingFace token from {token_file}: {e}")
        return None


# Default configuration instance for convenience
default_config = Config()
