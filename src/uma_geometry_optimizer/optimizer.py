"""Core geometry optimization module using Fairchem UMA models.

This module contains optimization logic for single structures and
batches of structures (e.g., conformer ensembles).
"""
from typing import List, Optional, TYPE_CHECKING

import logging
import torch_sim
from ase import Atoms
from ase.optimize import BFGS

from .config import Config, load_config_from_file
from .models import load_model_fairchem, load_model_torchsim, _check_device
from .decorators import time_it
from .structure import Structure

if TYPE_CHECKING:
    from fairchem.core import FAIRChemCalculator

logger = logging.getLogger(__name__)


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

    symbols = structure.symbols
    coordinates = structure.coordinates
    charge = structure.charge
    multiplicity = structure.multiplicity

    try:
        if not calculator:
            calculator = load_model_fairchem(config)

        atoms = Atoms(symbols=symbols, positions=coordinates)
        atoms.calc = calculator
        atoms.info = {'charge': charge, 'spin': multiplicity}

        logger.info("Starting single geometry optimization for structure with %d atoms", len(symbols))
        optimizer = BFGS(atoms, logfile=None)
        optimizer.run()
        logger.info("Optimization completed after %d steps", optimizer.get_number_of_steps())

        optimized_coords = atoms.get_positions().tolist()
        potential_energy = atoms.get_potential_energy()

        structure.coordinates = optimized_coords
        structure.energy = float(potential_energy)
        return structure

    except Exception as e:
        raise RuntimeError(f"Optimization failed: {str(e)}")


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

    for i, s in enumerate(structures):
        if s.n_atoms != len(s.coordinates):
            raise ValueError(f"Structure {i}: symbols/coords length mismatch")
        if s.n_atoms == 0:
            raise ValueError(f"Structure {i}: empty structure")

    device = _check_device(config.optimization.device)
    force_cpu = device == "cpu"

    logger.info("Optimization device: %s", 'CPU' if force_cpu else 'GPU')
    mode = config.optimization.batch_optimization_mode.lower()
    if mode == "sequential" or force_cpu:
        if not force_cpu and mode=="batch":
            logger.warning("Batch optimization mode requires GPU, falling back to sequential mode on CPU.")
        return _optimize_batch_sequential(structures, config)
    elif mode == "batch" and not force_cpu:
        return _optimize_batch_structures(structures, config)
    else:
        raise ValueError(f"Unknown optimization mode: {mode} on {config.optimization.device}. Use 'sequential' or 'batch', where batch is only supported on GPU.")


@time_it
def _optimize_batch_sequential(
    structures: List[Structure],
    config,
) -> List[Structure]:
    """Optimize structures sequentially using single geometry optimization."""
    calculator = load_model_fairchem(config)

    optimized_results: List[Structure] = []
    logger.info("Starting sequential optimization of %d structures", len(structures))

    for i, s in enumerate(structures):
        try:
            optimized = optimize_single_structure(s, config, calculator)
            optimized_results.append(optimized)
        except Exception as e:
            logger.warning("Structure %d optimization failed: %s", i+1, e)
            continue

    logger.info("Sequential optimization completed. %d/%d successful", len(optimized_results), len(structures))
    return optimized_results


@time_it
def _optimize_batch_structures(
    structures: List[Structure],
    config,
) -> List[Structure]:
    """Optimize structures in batches using Fairchem's batch prediction."""
    import torch
    import torch_sim as torchsim
    from torch_sim.optimizers import gradient_descent, fire
    from torch_sim.autobatching import InFlightAutoBatcher
    logger.info("Starting batch optimization of %d structures", len(structures))

    device = _check_device(config.optimization.device)
    model = load_model_torchsim(config)

    optimizer_name = getattr(config.optimization, 'batch_optimizer', 'fire') or 'fire'
    optimizer_name = str(optimizer_name).strip().lower()
    optimizer_fn = fire if optimizer_name == 'fire' else gradient_descent
    convergence_fn = torch_sim.generate_energy_convergence_fn(energy_tol=1e-6)

    ase_structures = [
        Atoms(symbols=s.symbols,
              positions=[tuple(c) for c in s.coordinates],
              info={'charge': s.charge, 'spin': s.multiplicity}
        ) for s in structures
    ]
    batched_state = torchsim.io.atoms_to_state(ase_structures, device=device, dtype=torch.float32)

    batcher = InFlightAutoBatcher(
        model,
        memory_scales_with="n_atoms",
        max_memory_padding=0.95,
        max_atoms_to_try=len(structures) * 2,
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
        s = Structure(
            symbols=atoms.get_chemical_symbols(),
            coordinates=atoms.get_positions().tolist(),
            energy=float(final_state.energy[i].item()),
            charge=int(structures[i].charge), # todo this is a workaround as long as charge and multiplicity not in torchsim
            multiplicity=int(structures[i].multiplicity),# todo this is a workaround as long as charge and multiplicity not in torchsim
            comment=f"Optimized with model {getattr(model, 'model_name', None) or ''} in batch mode",
        )
        results.append(s)
    return results
