"""uma_geometry_optimizer: A minimal package for molecular geometry optimization using Fairchem's UMA models.

This package provides essential tools for:
- Single molecule optimization from SMILES strings or XYZ files
- Batch optimization of multiple structures (e.g., conformer ensembles)
- Format conversion between SMILES and XYZ coordinates
- Configurable optimization parameters through JSON/YAML configuration files

The package is designed to be simple and focused, providing only the functionality
that is actually implemented and tested.
"""

from .structure import Structure
from .io_handler import read_xyz, read_multi_xyz, read_xyz_directory, smiles_to_xyz, smiles_to_ensemble, save_xyz_file, save_multi_xyz
from .optimizer import optimize_single_structure, optimize_structure_batch
from .models import load_model_fairchem, load_model_torchsim
from .config import Config, default_config, load_config_from_file, save_config_to_file
from .decorators import time_it

# Convenience functions for common workflows
def optimize_single_smiles(smiles: str, output_file: str = None, config: Config = None) -> Structure:
    """Convenient function to optimize a single molecule from SMILES.

    Args:
        smiles: SMILES string of the molecule to optimize.
        output_file: Optional output XYZ file path.
        config: Optional configuration object.

    Returns:
        Structure: The optimized molecular structure.
    """
    structure = smiles_to_xyz(smiles)  # returns Structure with auto charge/multiplicity
    if not isinstance(structure, Structure):
        raise ValueError("smiles_to_xyz did not return a Structure")
    structure.comment = f"Optimized from SMILES: {smiles}"
    result = optimize_single_structure(structure, config)

    if output_file:
        save_xyz_file(result, output_file)

    return result

def optimize_single_xyz_file(input_file: str, output_file: str = None, config: Config = None, charge: int = 0, multiplicity: int = 1) -> Structure:
    """Convenient function to optimize a molecule from XYZ file.

    Args:
        input_file: Path to input XYZ file.
        output_file: Optional output XYZ file path.
        config: Optional configuration object.
        charge: Optional total charge for the system (default 0).
        multiplicity: Optional spin multiplicity (default 1).

    Returns:
        Structure: The optimized molecular structure.
    """
    structure = read_xyz(input_file, charge=charge, multiplicity=multiplicity)
    structure.comment = f"Optimized from: {input_file}"
    result = optimize_single_structure(structure, config)

    if output_file:
        save_xyz_file(result, output_file)

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
    if not isinstance(conformers, list) or (len(conformers) and not isinstance(conformers[0], Structure)):
        raise ValueError("smiles_to_ensemble did not return a list of Structure")
    results = optimize_structure_batch(conformers, config)

    if output_file:
        comments = [f"Optimized conformer {i+1} from SMILES: {smiles}" for i in range(len(results))]
        save_multi_xyz(results, output_file, comments)

    return results

__version__ = "0.3.0"
__author__ = "Niklas Hölter"
__email__ = "niklas.hoelter@uni-muenster.de"

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

    # Convenience functions
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
