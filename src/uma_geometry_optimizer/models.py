"""Model loading utilities for Fairchem UMA and torch-sim models."""

from __future__ import annotations

import os
import logging
import torch
from pathlib import Path
from typing import Optional
from .config import Config
from .decorators import time_it

def _load_hf_token_to_env(config: Config):
    """Load Huggingface token from config to environment variable."""
    hf_token = config.optimization.get_huggingface_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

def _check_device(device: str) -> str:
    """Check if the specified device is available, fallback to CPU if not."""
    if device != "cpu" and not torch.cuda.is_available():
        logging.warning(f"Specified device '{device}' is not available. Falling back to 'cpu'.")
        return "cpu"
    return device

def _verify_model_name_and_cache_dir(config: Config) -> tuple[str, Optional[Path]]:
    """Verify that model name is provided and return model name and cache directory."""
    opt = config.optimization
    model_name = opt.model_name
    if not model_name:
        raise ValueError("Model name must be specified in the configuration")
    model_cache_dir = Path(opt.model_cache_dir) if opt.model_cache_dir else None
    if model_cache_dir and not model_cache_dir.exists():
        try:
            os.makedirs(model_cache_dir, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Could not create model cache directory '{model_cache_dir}': {e}") from e

    if not model_cache_dir.exists():
        model_cache_dir = None

    return model_name, model_cache_dir

@time_it
def load_model_fairchem(config: Config):
    """Load a FAIRChemCalculator using fairchem's pretrained_mlip helper."""
    from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore

    opt = config.optimization

    _load_hf_token_to_env(config)

    model_name, model_dir = _verify_model_name_and_cache_dir(config)
    device = _check_device(opt.device.lower())

    if model_dir:
        predictor = pretrained_mlip.get_predict_unit(
            model_name,
            device=device,
            cache_dir=model_dir,
        )
    else:
        predictor = pretrained_mlip.get_predict_unit(
            model_name,
            device=device,
        )

    calculator = FAIRChemCalculator(predict_unit=predictor, task_name="omol")
    return calculator

@time_it
def load_model_torchsim(config: Config):
    """Load a torch-sim FairChemModel from name or checkpoint path."""
    import torch_sim as torchsim  # type: ignore
    from torch_sim.models.fairchem import FairChemModel  # type: ignore

    opt = config.optimization
    force_cpu = _check_device(opt.device.lower()) == "cpu"
    model_name, model_path = _verify_model_name_and_cache_dir(config)
    _load_hf_token_to_env(config)

    if model_path:
        model = FairChemModel(model_name=model_name, model_cache_dir=model_path, cpu=force_cpu, task_name="omol")
        return model

    model = FairChemModel(model_name=model_name, cpu=force_cpu, task_name="omol")
    return model


