"""Core geometry optimization module using Fairchem UMA models.

This module contains optimization logic for single structures and
conformer ensembles (multiple conformers of the same molecule).
"""
from pathlib import Path
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.optimize import BFGS

from .config import Config, load_config_from_file
from .model import load_model

if TYPE_CHECKING:  # only for type hints, avoids runtime import of fairchem
    from fairchem.core import FAIRChemCalculator


def optimize_geometry(
    symbols: List[str],
    coordinates: List[List[float]],
    config: Optional[Config] = None,
    calculator: 'FAIRChemCalculator' = None,
) -> Tuple[List[str], List[List[float]], float]:
    """Optimize single molecular geometry using a pre-trained Fairchem UMA model.

    Args:
        symbols: List of atomic symbols (e.g., ['C', 'H', 'H', 'H']).
        coordinates: List of atomic coordinates, where each coordinate is [x, y, z].
        config: Configuration object. If None, loads from default config file.
        calculator: Calculator object. If None, loads model from config.

    Returns:
        A tuple containing:
            - List of atom symbols (unchanged from input)
            - List of optimized coordinates, where each coordinate is [x, y, z]
            - Final energy of the optimized structure in optimiizer units
    """
    if config is None:
        config = load_config_from_file()

    opt_config = config.optimization

    # Input validation, numpy conversion and checks
    if len(symbols) != len(coordinates):
        raise ValueError(f"Number of symbols ({len(symbols)}) must match number of coordinates ({len(coordinates)})")
    if len(symbols) == 0:
        raise ValueError("Cannot optimize empty structure")
    try:
        coords_array = np.array(coordinates, dtype=float)
        if coords_array.shape[1] != 3:
            raise ValueError("Each coordinate must have exactly 3 components (x, y, z)")
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid coordinate format: {e}")

    try:
        # Create ASE Atoms object
        atoms = Atoms(symbols=symbols, positions=coords_array.tolist())

        # Load the Fairchem model from model.py
        if not calculator:
            calculator = load_model(config)

        # Set the calculator for the atoms object
        atoms.calc = calculator

        # Set up optimizer with configuration parameters
        optimizer = BFGS(atoms, logfile=None)

        if opt_config.verbose:
            print(f"Starting geometry optimization for molecule with {len(symbols)} atoms")

        optimizer.run()

        if opt_config.verbose:
            print(f"Optimization completed after {optimizer.get_number_of_steps()} steps")

        # Extract optimized coordinates
        optimized_coords = atoms.get_positions().tolist()
        potential_energy = atoms.get_potential_energy()

        # Return symbols (unchanged), optimized coordinates and final energy
        return symbols, optimized_coords, potential_energy

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, ImportError)):
            raise
        else:
            raise RuntimeError(f"Optimization failed: {str(e)}")


def optimize_conformer_ensemble(
    conformers: List[Tuple[List[str], List[List[float]]]],
    config: Optional[Config] = None,
) -> List[Tuple[List[str], List[List[float]], float]]:
    """Optimize a list of conformers and return optimized structures with energies.

    This implementation supports two modes controlled by config.optimization.batch_optimization_mode:
    - "sequential": optimize each conformer with ASE BFGS and a shared calculator
    - "batch": use torch-sim batch optimization with a FairChem model wrapper

    Args:
        conformers: List of conformers, each a tuple (symbols, coordinates).
        config: Optional Config object.

    Returns:
        List of tuples: (symbols, optimized_coordinates, energy)
    """
    if config is None:
        config = load_config_from_file()

    if not conformers:
        return []

    # Basic validation: all conformers must have same atom count and non-empty
    first_count = len(conformers[0][0])
    for i, (symbols, coords) in enumerate(conformers):
        if len(symbols) != len(coords):
            raise ValueError(f"Conformer {i}: symbols/coords length mismatch")
        if len(symbols) == 0:
            raise ValueError(f"Conformer {i}: empty structure")
        if len(symbols) != first_count:
            raise ValueError("All conformers must have same number of atoms")

    if config.optimization.batch_optimization_mode.lower() == "sequential":
        return _optimize_sequential(conformers, config)
    elif config.optimization.batch_optimization_mode.lower() == "batch":
        return _optimize_batch(conformers, config)
    else:
        raise ValueError(f"Unknown optimization mode: {config.optimization.batch_optimization_mode.lower()}. Use 'sequential' or 'batch'")


def _optimize_sequential(
    conformers: List[Tuple[List[str], List[List[float]]]],
    config,
) -> List[Tuple[List[str], List[List[float]], float]]:
    """Optimize conformers sequentially using single geometry optimization."""

    calculator = load_model(config)
    opt_config = config.optimization

    optimized_results = []
    if opt_config.verbose:
        print(f"Starting sequential optimization of {len(conformers)} conformers")

    for i, (symbols, coords) in enumerate(conformers):
        try:
            opt_symbols, opt_coords, energy = optimize_geometry(symbols, coords, config, calculator)
            optimized_results.append((opt_symbols, opt_coords, energy))
        except Exception as e:
            if opt_config.verbose:
                print(f"Warning: Conformer {i+1} optimization failed: {e}")
            continue

    if opt_config.verbose:
        print(f"Sequential optimization completed. {len(optimized_results)}/{len(conformers)} successful")

    return optimized_results


def _optimize_batch(
    conformers: List[Tuple[List[str], List[List[float]]]],
    config,
) -> List[Tuple[List[str], List[List[float]], float]]:
    
    """Optimize conformers in batches using Fairchem's batch prediction."""

    # Defer heavy imports to runtime to avoid import-time dependency failures
    import torch
    import torch_sim as torchsim
    from torch_sim.models.fairchem import FairChemModel
    from torch_sim import optimizers as ts_optimizers

    opt_config = config.optimization
    force_cpu = not opt_config.device.lower().__contains__("cuda")


    model_path = Path(opt_config.model_path) if opt_config.model_path else None
    model_name = opt_config.model_name if opt_config.model_path is None else None
    if model_name is None and model_path is None:
        print("No model_name or model_path specified in config, defaulting to 'uma-s-1p1' and './models'")
        model_name = "uma-s-1p1"

    model = FairChemModel(model=model_path, model_name=model_name, cpu=force_cpu, task_name="omol")

    # Ensure positions are sequences of tuples for type checkers and ASE
    ase_conformers = [Atoms(symbols=symbols, positions=[tuple(c) for c in coords]) for symbols, coords in conformers]
    state = torchsim.io.atoms_to_state(ase_conformers, device=opt_config.device, dtype=torch.float32)

    # Select optimizer function based on configuration; default to FIRE
    optimizer_name = getattr(opt_config, 'batch_optimizer', 'fire') or 'fire'
    optimizer_name = str(optimizer_name).strip().lower()
    optimizer_fn = ts_optimizers.fire if optimizer_name == 'fire' else ts_optimizers.gradient_descent

    final_state = torchsim.optimize(
        system=state,
        model=model,
        optimizer=optimizer_fn
    )

    final_atoms = final_state.to_atoms()
    energy_results = model(final_state)

    results = []
    for i, atoms in enumerate(final_atoms):
        symbols = atoms.get_chemical_symbols()
        coords = atoms.get_positions().tolist()
        energy = float(energy_results["energy"][i].item())
        results.append((symbols, coords, energy))
    return results
