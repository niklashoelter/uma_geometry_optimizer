"""uma_geometry_optimizer: A minimal package for molecular geometry optimization using Fairchem's UMA models.

This package provides essential tools for:
- Single molecule optimization from SMILES strings or XYZ files
- Batch optimization of multiple structures (e.g., conformer ensembles)
- Format conversion between SMILES and XYZ coordinates
- Configurable optimization parameters through JSON/YAML configuration files

The package is designed to be simple and focused, providing only the functionality
that is actually implemented and tested.
"""

import logging
import sys

# Attach a NullHandler so library users don't get "No handler could be found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Ensure that if no handlers are configured, Python's lastResort logs to stdout
if isinstance(getattr(logging, "lastResort", None), logging.Handler):
    try:
        # lastResort is a StreamHandler; move its stream to stdout
        logging.lastResort.setStream(sys.stdout)  # type: ignore[attr-defined]
    except Exception:
        pass

from .structure import Structure
from .io_handler import (
    read_xyz,
    read_multi_xyz,
    read_xyz_directory,
    smiles_to_xyz,
    smiles_to_ensemble,
    save_xyz_file,
    save_multi_xyz,
)
from .optimizer import optimize_single_structure, optimize_structure_batch
from .models import load_model_fairchem, load_model_torchsim
from .config import Config, default_config, load_config_from_file, save_config_to_file
from .decorators import time_it
from .api import (
    optimize_single_smiles,
    optimize_single_xyz_file,
    optimize_smiles_ensemble,
)


__all__ = [
    # Data types
    "Structure",

    # I/O functions
    "read_xyz",
    "read_multi_xyz",
    "read_xyz_directory",
    "smiles_to_xyz",
    "smiles_to_ensemble",
    "save_xyz_file",
    "save_multi_xyz",

    # Optimization functions
    "optimize_single_structure",
    "optimize_structure_batch",

    # Convenience functions (re-exported from ``api``)
    "optimize_single_smiles",
    "optimize_single_xyz_file",
    "optimize_smiles_ensemble",

    # Model functions
    "load_model_torchsim",
    "load_model_fairchem",

    # Configuration
    "Config",
    "default_config",
    "load_config_from_file",
    "save_config_to_file",

    # Decorators
    "time_it",
]