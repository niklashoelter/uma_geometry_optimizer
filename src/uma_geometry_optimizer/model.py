"""Model management module for Fairchem UMA models.

This module handles all model-related operations including loading, downloading from
HuggingFace, and model configuration. It provides a unified interface for accessing
pre-trained Fairchem UMA models with proper error handling and validation.
"""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fairchem.core import FAIRChemCalculator

from .config import Config


def load_model(config: Config) -> 'FAIRChemCalculator':
    """Load Fairchem UMA model from local cache or download if needed.

    This function handles the complete model lifecycle:
    - Validates model configuration parameters
    - Sets up HuggingFace authentication if token is provided
    - Creates model cache directory if it doesn't exist
    - Downloads model from HuggingFace hub if not cached locally
    - Configures model with CPU/GPU settings from config
    - Returns ready-to-use calculator instance

    Returns:
        Configured FAIRChemCalculator instance ready for geometry optimization.

    Raises:
        ImportError: If fairchem library is not available or improperly installed.
        ValueError: If model name is invalid or not found in registry.
        RuntimeError: If model loading fails due to network, disk, or memory issues.
        OSError: If model cache directory cannot be created or accessed.
    """

    try:
        # Import here to provide better error messages
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as e:
        raise ImportError(
            "Fairchem library not found. Please install with: "
            "pip install uma-geometry-optimizer[fairchem] or pip install fairchem"
        ) from e

    opt_config = config.optimization

    # Validate model name
    if not opt_config.model_name or not opt_config.model_name.strip():
        raise ValueError("model_name cannot be empty")

    model_name = opt_config.model_name.strip()

    # Set up HuggingFace authentication if token is available
    hf_token = opt_config.get_huggingface_token()
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
        if opt_config.verbose:
            print("HuggingFace token configured for model download")

    # Create and validate model cache directory
    try:
        model_dir = opt_config.model_path
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
        else :
            model_dir = os.path.join(os.path.expanduser("~"), ".cache", "fairchem")
            os.makedirs(model_dir, exist_ok=True)
        if opt_config.verbose:
            print(f"Using model cache directory: {model_dir}")
    except OSError as e:
        raise OSError(f"Cannot create model cache directory {opt_config.model_path}: {e}") from e

    # Determine device
    device = opt_config.device

    if opt_config.verbose:
        print(f"Loading model '{model_name}' on device: {device}")

    try:
        # Load the model using Fairchem's pretrained model loader
        predictor = pretrained_mlip.get_predict_unit(
            model_name,
            device=device,
            cache_dir=model_dir
        )

        # Create calculator with proper task configuration
        calculator = FAIRChemCalculator(predict_unit=predictor, task_name="omol")

        if opt_config.verbose:
            print(f"Successfully loaded model '{model_name}'")

        return calculator

    except Exception as e:
        # Provide specific error messages for common issues
        error_msg = str(e).lower()

        if "not found" in error_msg or "404" in error_msg:
            raise ValueError(
                f"Model '{model_name}' not found in Fairchem registry. "
                "Please check the model name and ensure it's a valid UMA model."
            ) from e
        elif "connection" in error_msg or "network" in error_msg:
            raise RuntimeError(
                f"Network error while downloading model '{model_name}'. "
                "Please check your internet connection and try again."
            ) from e
        elif "memory" in error_msg or "cuda" in error_msg:
            raise RuntimeError(
                f"Memory or CUDA error while loading model '{model_name}'. "
            ) from e
        else:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e


def validate_model_availability(model_name: str) -> bool:
    """Check if a model is available in the Fairchem registry.

    Args:
        model_name: Name of the model to check.

    Returns:
        True if model is available, False otherwise.

    Note:
        This function requires an active internet connection to query
        the HuggingFace model registry.
    """
    try:
        from fairchem.core import pretrained_mlip
        # This will raise an exception if model is not found
        pretrained_mlip.get_predict_unit(model_name, device="cpu")
        return True
    except Exception:
        return False
