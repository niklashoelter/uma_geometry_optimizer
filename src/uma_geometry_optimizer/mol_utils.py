"""Molecular utilities for SMILES processing and conformer generation.

This module provides functions for converting SMILES strings to 3D molecular
structures and generating conformer ensembles using the morfeus library with RDKit.
"""

from typing import List, Tuple


def smiles_to_conformer_ensemble(smiles: str, max_num_confs: int = 10) -> List[Tuple[List[str], List[List[float]]]]:
    """Generate multiple conformers from a SMILES string.

    This function uses the morfeus library to generate conformers from a SMILES string.
    The resulting conformers are automatically pruned based on RMSD and sorted by energy.

    Args:
        smiles: Valid SMILES string representing the molecular structure.
        max_num_confs: Maximum number of conformers to return (default: 10).

    Returns:
        List of conformers, each as a tuple of (element_symbols, coordinates).
        Element symbols are strings like ['C', 'H', 'O'].
        Coordinates are nested lists [[x1,y1,z1], [x2,y2,z2], ...].

    Raises:
        ValueError: If SMILES string is invalid or conformer generation fails.
        ImportError: If morfeus or RDKit dependencies are not available.

    Example:
        >>> conformers = smiles_to_conformer_ensemble("CCO", 5)
        >>> print(f"Generated {len(conformers)} conformers")
        >>> symbols, coords = conformers[0]  # First (lowest energy) conformer
        >>> print(f"First conformer has {len(symbols)} atoms")

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
        # Lazy import to avoid import-time dependency failures
        from morfeus.conformer import ConformerEnsemble  # type: ignore

        # Generate conformer ensemble using morfeus
        ensemble = ConformerEnsemble.from_rdkit(smiles.strip())
        ensemble.prune_rmsd()
        ensemble.sort()

        # Extract conformers up to the requested maximum
        conformers = []
        for i, conformer in enumerate(ensemble):
            if i >= max_num_confs:
                break

            # Get atomic elements and coordinates
            atoms = conformer.elements
            coordinates = conformer.coordinates

            # Validate data consistency
            if len(atoms) != len(coordinates):
                continue  # Skip invalid conformers

            conformers.append((atoms, coordinates))

        if not conformers:
            raise ValueError("No valid conformers could be generated from SMILES")

        return conformers

    except ImportError as e:
        raise ImportError(
            "Required dependencies not found. Please install with: "
            "uv pip install 'uma-geometry-optimizer' or install 'morfeus-ml rdkit'"
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to generate conformers from SMILES '{smiles}': {e}") from e


def smiles_to_structure(smiles: str) -> Tuple[List[str], List[List[float]]]:
    """Convert SMILES string to a single 3D molecular structure.

    This function generates the lowest-energy conformer from a SMILES string.
    It internally uses smiles_to_conformer_ensemble and returns only the first
    (lowest energy) conformer.

    Args:
        smiles: Valid SMILES string representing the molecular structure.

    Returns:
        Tuple of (element_symbols, coordinates) for the lowest-energy conformer.
        Element symbols are strings like ['C', 'H', 'O'].
        Coordinates are nested lists [[x1,y1,z1], [x2,y2,z2], ...].

    Raises:
        ValueError: If SMILES string is invalid or structure generation fails.
        ImportError: If morfeus or RDKit dependencies are not available.

    Example:
        >>> symbols, coords = smiles_to_structure("CCO")
        >>> print(f"Ethanol has {len(symbols)} atoms")
        >>> print(f"First atom is {symbols[0]} at {coords[0]}")

    Note:
        This function is equivalent to calling smiles_to_conformer_ensemble
        with max_num_confs=1 and extracting the first result.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string cannot be empty")

    try:
        # Generate ensemble and take the first (lowest energy) conformer
        ensemble = smiles_to_conformer_ensemble(smiles.strip(), max_num_confs=1)

        if not ensemble:
            raise ValueError("No conformers generated from SMILES")

        return ensemble[0]  # Return first conformer (lowest energy)

    except Exception as e:
        if isinstance(e, (ValueError, ImportError)):
            raise
        else:
            raise ValueError(f"Failed to generate structure from SMILES '{smiles}': {e}") from e
