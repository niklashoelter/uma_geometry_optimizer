"""Molecular utilities for SMILES processing and structure/conformer generation.

This module provides functions for converting SMILES strings to 3D molecular
structures and generating conformer ensembles using the morfeus library with RDKit.
"""

from typing import List, Tuple
from numbers import Integral
from .decorators import time_it
from .structure import Structure


def _to_symbol_list(elements) -> List[str]:
    """Convert a sequence of element descriptors to a list of atomic symbols.

    Accepts element symbols as strings or atomic numbers (ints); converts to
    a list of string symbols. Also supports numpy arrays transparently.
    """
    from ase.data import chemical_symbols

    try:
        if hasattr(elements, "tolist"):
            elements = elements.tolist()
    except Exception:
        pass

    symbols: List[str] = []
    for e in elements:
        if isinstance(e, str):
            symbols.append(e)
        elif isinstance(e, Integral):
            try:
                symbols.append(chemical_symbols[int(e)])
            except Exception:
                raise ValueError(f"Invalid atomic number: {e}")
        else:
            symbols.append(str(e))
    return symbols


def _to_coord_list(coords) -> List[Tuple[float, float, float]]:
    """Convert coordinates to a nested Python list of float tuples."""
    try:
        if hasattr(coords, "tolist"):
            coords = coords.tolist()
    except Exception:
        pass
    return [(
        float(row[0]), float(row[1]), float(row[2])
    ) for row in coords]

@time_it
def smiles_to_conformer_ensemble(smiles: str, max_num_confs: int = 10) -> List[Structure]:
    """Generate multiple conformers from a SMILES string.

    This function uses the morfeus library to generate conformers from a SMILES string.
    The resulting conformers are automatically pruned based on RMSD and sorted by energy.

    Args:
        smiles: Valid SMILES string representing the molecular structure.
        max_num_confs: Maximum number of conformers to return (default: 10).

    Returns:
        List of Structure objects representing conformers.

    Raises:
        ValueError: If SMILES string is invalid or conformer generation fails.
        ImportError: If morfeus or RDKit dependencies are not available.

    Note:
        Conformers are sorted by energy (lowest first) and pruned by RMSD
        to remove duplicates. The actual number returned may be less than
        max_num_confs if fewer unique conformers are generated.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string cannot be empty")

    if max_num_confs <= 0:
        raise ValueError("max_num_confs must be positive")

    try:
        from morfeus.conformer import ConformerEnsemble  # type: ignore

        ensemble = ConformerEnsemble.from_rdkit(smiles.strip())
        ensemble.prune_rmsd()
        ensemble.sort()

        structures: List[Structure] = []
        for i, conformer in enumerate(ensemble):
            if i >= max_num_confs:
                break

            atoms = _to_symbol_list(getattr(conformer, "elements", []))
            coordinates = _to_coord_list(getattr(conformer, "coordinates", []))

            if len(atoms) != len(coordinates):
                continue

            structures.append(Structure(symbols=atoms, coordinates=coordinates, charge=ensemble.charge, multiplicity=ensemble.multiplicity))

        if not structures:
            raise ValueError("No valid conformers could be generated from SMILES")

        return structures

    except ImportError as e:
        raise ImportError(
            "Required dependencies not found. Please install with: "
            "uv pip install 'uma-geometry-optimizer' or install 'morfeus-ml rdkit'"
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to generate conformers from SMILES '{smiles}': {e}") from e


def smiles_to_structure(smiles: str) -> Structure:
    """Convert SMILES string to a single 3D molecular structure.

    This function generates the lowest-energy conformer from a SMILES string.
    It internally uses smiles_to_conformer_ensemble and returns only the first
    (lowest energy) conformer.

    Args:
        smiles: Valid SMILES string representing the molecular structure.

    Returns:
        Structure instance for the lowest-energy conformer.

    Raises:
        ValueError: If SMILES string is invalid or structure generation fails.
        ImportError: If morfeus or RDKit dependencies are not available.

    Note:
        This function is equivalent to calling smiles_to_conformer_ensemble
        with max_num_confs=1 and extracting the first result.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string cannot be empty")

    try:
        ensemble = smiles_to_conformer_ensemble(smiles.strip(), max_num_confs=1)

        if not ensemble:
            raise ValueError("No conformers generated from SMILES")

        return ensemble[0]

    except Exception as e:
        if isinstance(e, (ValueError, ImportError)):
            raise
        else:
            raise ValueError(f"Failed to generate structure from SMILES '{smiles}': {e}") from e
