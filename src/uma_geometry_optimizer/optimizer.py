"""Core geometry optimization module using Fairchem UMA models.

This module contains optimization logic for single structures and
batches of structures (e.g., conformer ensembles).
"""
from pathlib import Path
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np
import torch_sim
from ase import Atoms
from ase.optimize import BFGS

from .config import Config, load_config_from_file
from .model import load_model
from .decorators import time_it
from .structure import Structure

if TYPE_CHECKING:  # only for type hints, avoids runtime import of fairchem
    from fairchem.core import FAIRChemCalculator


@time_it
def optimize_single_structure(
    structure: Structure,
    config: Optional[Config] = None,
    calculator: 'FAIRChemCalculator' = None,
) -> Structure:
    """Optimize a single Structure using a pre-trained Fairchem UMA model.

    Returns the same Structure with optimized coordinates and energy set.
    """
    if config is None:
        config = load_config_from_file()

    opt_config = config.optimization

    symbols = structure.symbols
    coordinates = structure.coordinates

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
            print(f"Starting geometry optimization for structure with {len(symbols)} atoms")

        optimizer.run()

        if opt_config.verbose:
            print(f"Optimization completed after {optimizer.get_number_of_steps()} steps")

        # Extract optimized coordinates
        optimized_coords = atoms.get_positions().tolist()
        potential_energy = atoms.get_potential_energy()

        # Return updated Structure
        structure.coordinates = optimized_coords
        structure.energy = float(potential_energy)
        return structure

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, ImportError)):
            raise
        else:
            raise RuntimeError(f"Optimization failed: {str(e)}")


@time_it
def optimize_single_geometry(
    symbols: List[str],
    coordinates: List[List[float]],
    config: Optional[Config] = None,
    calculator: 'FAIRChemCalculator' = None,
) -> Tuple[List[str], List[List[float]], float]:
    """Backward-compatible wrapper to optimize a single structure from symbols/coords."""
    struct = Structure(symbols=symbols, coordinates=coordinates)
    optimized = optimize_single_structure(struct, config, calculator)
    return optimized.symbols, optimized.coordinates, optimized.energy if optimized.energy is not None else 0.0


def optimize_structure_batch(
    structures: List[Structure],
    config: Optional[Config] = None,
) -> List[Structure]:
    """Optimize a list of structures and return optimized structures with energies.

    Supports two modes controlled by config.optimization.batch_optimization_mode:
    - "sequential": optimize each structure with ASE BFGS and a shared calculator
    - "batch": use torch-sim batch optimization with a FairChem model wrapper
    """
    if config is None:
        config = load_config_from_file()

    if not structures:
        return []

    # Basic validation: all structures must have same atom count and non-empty
    first_count = structures[0].n_atoms
    for i, s in enumerate(structures):
        if s.n_atoms != len(s.coordinates):
            raise ValueError(f"Structure {i}: symbols/coords length mismatch")
        if s.n_atoms == 0:
            raise ValueError(f"Structure {i}: empty structure")
        if s.n_atoms != first_count:
            raise ValueError("All structures must have same number of atoms")

    mode = config.optimization.batch_optimization_mode.lower()
    if mode == "sequential":
        return _optimize_sequential_structures(structures, config)
    elif mode == "batch":
        return _optimize_batch_structures(structures, config)
    else:
        raise ValueError(f"Unknown optimization mode: {mode}. Use 'sequential' or 'batch'")


@time_it
def _optimize_sequential_structures(
    structures: List[Structure],
    config,
) -> List[Structure]:
    """Optimize structures sequentially using single geometry optimization."""
    calculator = load_model(config)
    opt_config = config.optimization

    optimized_results: List[Structure] = []
    if opt_config.verbose:
        print(f"Starting sequential optimization of {len(structures)} structures")

    for i, s in enumerate(structures):
        try:
            optimized = optimize_single_structure(s, config, calculator)
            optimized_results.append(optimized)
        except Exception as e:
            if opt_config.verbose:
                print(f"Warning: Structure {i+1} optimization failed: {e}")
            continue

    if opt_config.verbose:
        print(f"Sequential optimization completed. {len(optimized_results)}/{len(structures)} successful")

    return optimized_results


@time_it
def _optimize_batch_structures(
    structures: List[Structure],
    config,
) -> List[Structure]:
    """Optimize structures in batches using Fairchem's batch prediction."""

    # Defer heavy imports to runtime to avoid import-time dependency failures
    import torch
    import torch_sim as torchsim
    from torch_sim.models.fairchem import FairChemModel
    from torch_sim.optimizers import gradient_descent, fire
    from torch_sim.autobatching import InFlightAutoBatcher

    opt_config = config.optimization
    force_cpu = not opt_config.device.lower().__contains__("cuda")

    model_path = Path(opt_config.model_path) if opt_config.model_path else None
    model_name = opt_config.model_name if opt_config.model_path is None else None
    if model_name is None and model_path is None:
        print("No model_name or model_path specified in config, defaulting to 'uma-s-1p1' and './models'")
        model_name = "uma-s-1p1"
    print(f"Loading model '{model_name}' from '{model_path or './models'}' on {'CPU' if force_cpu else 'GPU'}")
    model = FairChemModel(model=model_path, model_name=model_name, cpu=force_cpu, task_name="omol")

    optimizer_name = getattr(opt_config, 'batch_optimizer', 'fire') or 'fire'
    optimizer_name = str(optimizer_name).strip().lower()
    optimizer_fn = fire if optimizer_name == 'fire' else gradient_descent

    convergence_fn = torch_sim.generate_energy_convergence_fn(energy_tol=1e-6)

    # Ensure positions are sequences of tuples for type checkers and ASE
    ase_structures = [Atoms(symbols=s.symbols, positions=[tuple(c) for c in s.coordinates]) for s in structures]
    batched_state = torchsim.io.atoms_to_state(ase_structures, device=opt_config.device, dtype=torch.float32)

    batcher = InFlightAutoBatcher(
        model,
        memory_scales_with="n_atoms",
        max_memory_padding=0.95,
        max_atoms_to_try=len(structures) * 3, # todo this is not the number of atoms!! there is a current bug in torchsim
    )

    final_state = torch_sim.runners.optimize(
        system = batched_state,
        model = model,
        optimizer = optimizer_fn,
        convergence_fn = convergence_fn,
        autobatcher = batcher,
        steps_between_swaps = 3
    )

    final_atoms = final_state.to_atoms()

    results: List[Structure] = []
    for i, atoms in enumerate(final_atoms):
        symbols = atoms.get_chemical_symbols()
        coords = atoms.get_positions().tolist()
        energy = float(final_state.energy[i].item())
        s = structures[i]
        s.symbols = symbols
        s.coordinates = coords
        s.energy = energy
        results.append(s)
    return results


def optimize_conformer_ensemble(
    conformers: List[Tuple[List[str], List[List[float]]]],
    config: Optional[Config] = None,
) -> List[Tuple[List[str], List[List[float]], float]]:
    """Backward-compatible wrapper for optimizing conformer ensembles using tuples."""
    structures = [Structure(symbols=s, coordinates=c) for (s, c) in conformers]
    optimized = optimize_structure_batch(structures, config)
    return [(s.symbols, s.coordinates, float(s.energy) if s.energy is not None else 0.0) for s in optimized]

