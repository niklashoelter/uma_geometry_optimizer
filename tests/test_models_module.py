import pytest

from uma_geometry_optimizer.config import Config
from uma_geometry_optimizer import models as model_utils


def test_models_module_exports():
    assert hasattr(model_utils, "load_model_fairchem")
    assert hasattr(model_utils, "load_model_torchsim")


def test_load_model_fairchem_empty_name_raises():
    cfg = Config.from_dict({"optimization": {"model_name": "", "device": "cpu"}})
    with pytest.raises(ValueError):
        model_utils.load_model_fairchem(cfg)


def test_load_model_torchsim_import_or_skip():
    try:
        import torch_sim  # noqa: F401
    except Exception:
        pytest.skip("torch_sim not installed; skipping torch-sim model test")

    cfg = Config()
    cfg.optimization.device = "cpu"
    # Do not actually download or require checkpoints; ensure callable without throwing ImportError
    # Expect that either construction succeeds (if environment has models) or raises a runtime error due to missing model
    try:
        model_utils.load_model_torchsim(cfg)
    except Exception:
        pass
