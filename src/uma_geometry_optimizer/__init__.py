"""uma_geometry_optimizer: A minimal package for molecular geometry optimization using Fairchem's UMA models.

This package provides essential tools for:
- Single molecule optimization from SMILES strings or XYZ files
- Conformer ensemble optimization (sequential processing of multiple conformers)
- Format conversion between SMILES and XYZ coordinates
- Configurable optimization parameters through JSON/YAML configuration files

The package is designed to be simple and focused, providing only the functionality
that is actually implemented and tested.
"""

from .io_handler import read_xyz, read_multi_xyz, read_xyz_directory, smiles_to_xyz, smiles_to_ensemble, save_xyz_file, save_multi_xyz
from .optimizer import optimize_geometry, optimize_conformer_ensemble
from .model import load_model
from .config import Config, OptimizationConfig, default_config, load_config_from_file, save_config_to_file

# Convenience functions for common workflows
def optimize_smiles(smiles: str, output_file: str = None, config: Config = None) -> tuple:
    """Convenient function to optimize a single molecule from SMILES.

    Args:
        smiles: SMILES string of the molecule to optimize.
        output_file: Optional output XYZ file path.
        config: Optional configuration object.

    Returns:
        Tuple of (symbols, optimized_coordinates, energy).
    """
    symbols, coords = smiles_to_xyz(smiles)
    result = optimize_geometry(symbols, coords, config)

    if output_file:
        save_xyz_file(result[0], result[1], result[2], output_file, f"Optimized from SMILES: {smiles}")

    return result

def optimize_xyz_file(input_file: str, output_file: str = None, config: Config = None) -> tuple:
    """Convenient function to optimize a molecule from XYZ file.

    Args:
        input_file: Path to input XYZ file.
        output_file: Optional output XYZ file path.
        config: Optional configuration object.

    Returns:
        Tuple of (symbols, optimized_coordinates, energy).
    """
    symbols, coords = read_xyz(input_file)
    result = optimize_geometry(symbols, coords, config)

    if output_file:
        save_xyz_file(result[0], result[1], result[2], output_file, f"Optimized from: {input_file}")

    return result

def optimize_smiles_ensemble(smiles: str, num_conformers: int = 10, output_file: str = None, config: Config = None) -> list:
    """Convenient function to optimize conformer ensemble from SMILES.

    Args:
        smiles: SMILES string of the molecule.
        num_conformers: Number of conformers to generate and optimize.
        output_file: Optional output multi-XYZ file path.
        config: Optional configuration object.

    Returns:
        List of (symbols, optimized_coordinates, energy) tuples.
    """
    conformers = smiles_to_ensemble(smiles, num_conformers)
    results = optimize_conformer_ensemble(conformers, config)

    if output_file:
        comments = [f"Optimized conformer {i+1} from SMILES: {smiles}" for i in range(len(results))]
        save_multi_xyz(results, output_file, comments)

    return results

__version__ = "0.2.0"
__author__ = "Niklas Hölter"
__email__ = "niklas.hoelter@uni-muenster.de"

__all__ = [
    # I/O functions
    "read_xyz",
    "read_multi_xyz", 
    "read_xyz_directory",
    "smiles_to_xyz",
    "smiles_to_ensemble",
    "save_xyz_file",
    "save_multi_xyz",

    # Optimization functions
    "optimize_geometry",
    "optimize_conformer_ensemble",

    # Convenience functions
    "optimize_smiles",
    "optimize_xyz_file",
    "optimize_smiles_ensemble",

    # Model functions
    "load_model",

    # Configuration
    "Config",
    "OptimizationConfig",
    "default_config",
    "load_config_from_file",
    "save_config_to_file",
]
