import json
from pathlib import Path

from uma_geometry_optimizer.config import (
    Config,
    OptimizationConfig,
    load_config_from_file,
    save_config_to_file,
)


def test_config_defaults_and_validation(tmp_path: Path):
    cfg = Config()
    # Defaults
    assert cfg.optimization.batch_optimization_mode in {"batch", "sequential"}
    assert cfg.optimization.batch_optimizer in {"fire", "gradient_descent"}
    assert cfg.optimization.batch_optimizer == "fire"
    assert cfg.optimization.max_num_conformers > 0
    assert cfg.optimization.conformer_seed >= 0
    assert cfg.optimization.device in {"cpu", "cuda"}

    # Roundtrip save/load
    out = tmp_path / "config.json"
    save_config_to_file(cfg, str(out))
    reloaded = load_config_from_file(str(out))
    assert reloaded.optimization.batch_optimizer == cfg.optimization.batch_optimizer
    assert reloaded.optimization.device == cfg.optimization.device


def test_config_from_dict_custom_batch_optimizer():
    data = {
        "optimization": {
            "batch_optimization_mode": "batch",
            "batch_optimizer": "gradient_descent",
            "device": "cpu",
        }
    }
    cfg = Config.from_dict(data)
    assert cfg.optimization.batch_optimizer == "gradient_descent"


def test_json_schema_contains_batch_optimizer():
    # Ensure the test config used by the suite includes batch_optimizer
    root = Path(__file__).resolve().parents[1]
    config_json = root / "tests" / "test_config.json"
    with config_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "optimization" in data
    assert "batch_optimizer" in data["optimization"]
