import io
import json
import os
from pathlib import Path
import sys
import types
import builtins

import pytest

from uma_geometry_optimizer.config import (
    Config,
    load_config_from_file,
    save_config_to_file,
    get_huggingface_token,
    DEFAULT_CONFIG,
)


def test_config_defaults_and_attribute_access():
    cfg = Config()
    # default merge
    assert cfg.to_dict()["optimization"]["model_name"] == DEFAULT_CONFIG["optimization"]["model_name"]
    # attribute access and set
    assert isinstance(cfg.optimization.to_dict(), dict)
    cfg.optimization.device = "cpu"
    assert cfg.optimization.device == "cpu"


def test_load_and_save_config_json(tmp_path):
    # Save custom config
    cfg = Config()
    cfg.optimization.verbose = False
    out = tmp_path / "config.json"
    save_config_to_file(cfg, str(out))
    assert out.exists()

    # Load and merge
    loaded = load_config_from_file(str(out))
    assert isinstance(loaded, Config)
    assert loaded.optimization.verbose is False
    # unknown keys preserved
    data = loaded.to_dict()
    data["custom"] = {"a": 1}
    loaded2 = Config.from_dict(data)
    assert loaded2.to_dict()["custom"]["a"] == 1


def test_load_config_missing_returns_defaults(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    cfg = load_config_from_file(str(missing))
    assert isinstance(cfg, Config)


def test_get_huggingface_token_from_inline_and_file(tmp_path):
    token_file = tmp_path / "token.txt"
    token_file.write_text("secrettoken\n", encoding="utf-8")

    cfg = Config()
    cfg.optimization.huggingface_token = "inline"
    assert cfg.optimization.get_huggingface_token() == "inline"

    cfg2 = Config()
    cfg2.optimization.huggingface_token = None
    cfg2.optimization.huggingface_token_file = str(token_file)
    assert cfg2.optimization.get_huggingface_token() == "secrettoken"

    # dict helper
    d = {"optimization": {"huggingface_token_file": str(token_file)}}
    assert get_huggingface_token(d) == "secrettoken"


def test_get_huggingface_token_missing_file_prints_warning(tmp_path, capsys):
    cfg = Config()
    cfg.optimization.huggingface_token = None
    cfg.optimization.huggingface_token_file = str(tmp_path / "missing.txt")
    cfg.optimization.verbose = True
    token = cfg.optimization.get_huggingface_token()
    assert token is None
    out = capsys.readouterr().out
    assert "Could not read HuggingFace token" in out


def test_yaml_requires_pyyaml_when_used(tmp_path, monkeypatch):
    # Simulate ImportError when attempting to import yaml during save
    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return orig_import(name, *args, **kwargs)

    orig_import = builtins.__import__

    # Save YAML should raise ImportError
    p = tmp_path / "cfg.yaml"
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        save_config_to_file(Config(), str(p))

    # Load YAML should also raise ImportError
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        load_config_from_file(str(p))

