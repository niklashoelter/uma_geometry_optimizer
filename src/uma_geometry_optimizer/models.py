"""Model loading utilities for Fairchem UMA and torch-sim models."""

from __future__ import annotations

import os
import logging
import torch
from pathlib import Path
from typing import Optional
from .config import Config

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

def load_model_fairchem(config: Config):
    """Load a FAIRChemCalculator using fairchem's pretrained_mlip helper."""
    from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore

    opt = config.optimization

    _load_hf_token_to_env(config)

    model_name = (opt.model_name or "").strip()
    if not model_name:
        raise ValueError("model_name cannot be empty")

    try:
        model_dir = opt.model_path if opt.model_path else os.path.join(os.path.expanduser("~"), ".cache", "fairchem")
        os.makedirs(model_dir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Cannot create model cache directory {opt.model_path}: {e}") from e

    device = _check_device(opt.device.lower())

    try:
        predictor = pretrained_mlip.get_predict_unit(
            model_name,
            device=device,
            cache_dir=model_dir,
        )
        calculator = FAIRChemCalculator(predict_unit=predictor, task_name="omol")
        return calculator
    except Exception as e:
        msg = str(e).lower()
        if "not found" in msg or "404" in msg:
            raise ValueError(f"Model '{model_name}' not found in Fairchem registry") from e
        raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e


def load_model_torchsim(config: Config):
    """Load a torch-sim FairChemModel from name or checkpoint path."""
    import torch_sim as torchsim  # type: ignore
    from torch_sim.models.fairchem import FairChemModel  # type: ignore

    opt = config.optimization
    force_cpu = _check_device(opt.device.lower()) == "cpu"

    model: Optional[FairChemModel] = None
    model_path = Path(opt.model_path) if opt.model_path else None
    model_name = opt.model_name if opt.model_name else None

    _load_hf_token_to_env(config)

    if model_path:
        if not model_path.exists():
            logging.error(f"Specified model_path does not exist: {model_path}")
            logging.error("Falling back to default model 'uma-s-1p1'")
            model_name = "uma-s-1p1"
            model_path = None
        else:
            try:
                model = FairChemModel(model=model_path, cpu=force_cpu, task_name="omol")
            except Exception:
                logging.error(f"Failed to load model from path: {model_path}")
                logging.error("Falling back to default model 'uma-s-1p1'")
                model_name = "uma-s-1p1"
                model_path = None

    if model_name is None and model_path is None:
        logging.error("No model_name or model_path specified in config, defaulting to 'uma-s-1p1'")
        model_name = "uma-s-1p1"

    if not model:
        model = FairChemModel(model=model_path, model_name=model_name, cpu=force_cpu, task_name="omol")

    return model

