"""Input/Output handler for molecular structures.

This module provides functions for reading molecular structures from various formats
and converting between different representations.
"""

import os
import glob
from typing import List, Tuple, Optional

from .mol_utils import smiles_to_structure, smiles_to_conformer_ensemble

def read_xyz(file_path: str) -> Tuple[List[str], List[List[float]]]:
    """Read an XYZ file and return atom symbols and their coordinates.
    
    Args:
        file_path: Path to the XYZ file to read.
        
    Returns:
        A tuple containing:
            - List of atom symbols (e.g., ['C', 'H', 'H', 'H'])
            - List of coordinates, where each coordinate is [x, y, z]
            
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is invalid.
        
    Example:
        >>> symbols, coords = read_xyz("molecule.xyz")
        >>> print(symbols)
        ['C', 'H', 'H', 'H', 'H']
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    symbols = []
    coordinates = []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Preserve comment line even if empty. Only strip \n but keep blank lines
        lines = [line.rstrip('\n') for line in lines]

        # First line should contain the number of atoms
        try:
            num_atoms = int(lines[0])
        except ValueError:
            raise ValueError("First line must contain the number of atoms as an integer")

        # Second line is typically a comment line (skip it)
        # Do not drop empty comment line; atom lines start at index 2
        if len(lines) < 2 + num_atoms:
            raise ValueError(f"Expected {num_atoms} atom lines, but found {max(0, len(lines)-2)}")

        for i in range(num_atoms):
            line = lines[2 + i]
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Line {i+3} must contain at least 4 elements: symbol x y z")

            symbol = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                raise ValueError(f"Invalid coordinates in line {i+3}: {parts[1:4]}")

            symbols.append(symbol)
            coordinates.append([x, y, z])

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise ValueError(f"Error reading XYZ file: {str(e)}")

    return symbols, coordinates

def read_multi_xyz(file_path: str) -> List[Tuple[List[str], List[List[float]]]]:
    """Read an XYZ file containing multiple structures (conformers).

    Args:
        file_path: Path to the multi-structure XYZ file.

    Returns:
        List of structures, each a tuple of (symbols, coordinates).

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is invalid.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    structures = []

    try:
        with open(file_path, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]

        i = 0
        while i < len(lines):
            # Skip empty lines
            if lines[i].strip() == '':
                i += 1
                continue

            # Try to parse number of atoms
            try:
                num_atoms = int(lines[i].strip())
            except ValueError:
                i += 1
                continue

            # Ensure enough lines remain (including comment)
            if i + 1 + num_atoms >= len(lines):
                break

            # Comment line at i+1 (may be empty)
            symbols = []
            coordinates = []

            for j in range(num_atoms):
                parts = lines[i + 2 + j].split()
                if len(parts) < 4:
                    break
                symbol = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    break
                symbols.append(symbol)
                coordinates.append([x, y, z])

            if len(symbols) == num_atoms:
                structures.append((symbols, coordinates))

            i = i + 2 + num_atoms

    except Exception as e:
        raise ValueError(f"Error reading multi-XYZ file: {str(e)}")

    return structures

def read_xyz_directory(directory_path: str) -> List[Tuple[List[str], List[List[float]]]]:
    """Read all XYZ files from a directory.

    Args:
        directory_path: Path to directory containing XYZ files.

    Returns:
        List of structures from all XYZ files in the directory.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no valid XYZ files are found.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} not found")

    xyz_files = glob.glob(os.path.join(directory_path, "*.xyz"))

    if not xyz_files:
        raise ValueError(f"No XYZ files found in directory {directory_path}")

    structures = []

    for xyz_file in xyz_files:
        try:
            symbols, coordinates = read_xyz(xyz_file)
            structures.append((symbols, coordinates))
        except Exception as e:
            print(f"Warning: Failed to read {xyz_file}: {e}")

    if not structures:
        raise ValueError("No valid structures could be read from any XYZ files")

    return structures


def smiles_to_xyz(smiles_string: str, return_full_xyz_str = False) -> Tuple[List[str], List[List[float]]] | str:
    """Convert a SMILES string to XYZ file format string.

    This function takes a SMILES string as input and converts it into an XYZ
    file format string using the molecular utilities from mol_utils module.

    Args:
        smiles_string: A valid SMILES string representing the molecular structure.

    Returns:
        A string containing the molecule's coordinates in XYZ format.

    Raises:
        ValueError: If the SMILES string is invalid or conversion fails.
        RuntimeError: If there are issues with the molecular conversion process.

    Example:
        >>> xyz_string = smiles_to_xyz("CCO")
        >>> print(xyz_string)
        9
        Generated from SMILES: CCO
        C    -0.644300    0.133900   -0.120500
        ...
    """
    if not smiles_string or not smiles_string.strip():
        raise ValueError("SMILES string cannot be empty or None")

    try:
        # Use mol_utils function to get atoms and coordinates
        atoms, coordinates = smiles_to_structure(smiles_string.strip())

        if len(atoms) != len(coordinates):
            raise ValueError("Mismatch between number of atoms and coordinates")

        # Format as XYZ string
        if return_full_xyz_str:
            xyz_lines = [str(len(atoms))]
            xyz_lines.append(f"Generated from SMILES: {smiles_string}")

            for atom, coord in zip(atoms, coordinates):
                if len(coord) != 3:
                    raise ValueError("Invalid coordinate format - expected 3D coordinates")
                xyz_lines.append(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
            return "\n".join(xyz_lines)

        else:
            return atoms, coordinates

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise RuntimeError(f"Error converting SMILES to XYZ: {str(e)}")

def smiles_to_ensemble(smiles_string: str, num_conformers: int, return_full_xyz_str = False) -> List[str] | List[Tuple[List[str], List[List[float]]]]:
    """Generate a conformational ensemble from a SMILES string.

    This function takes a SMILES string and generates a specified number of
    conformers using the conformer generation utilities from mol_utils module.

    Args:
        smiles_string: A valid SMILES string representing the molecular structure.
        num_conformers: The maximum number of conformers to generate.
        return_full_xyz_str: If True, returns a list of XYZ format strings for each conformer.

    Returns:
        A list of strings, where each string is a conformer in XYZ format,
        or a list of (atoms, coordinates) tuples if return_full_xyz_str is False.

    Raises:
        ValueError: If the SMILES string is invalid, num_conformers is not positive,
                   or conformer generation fails.
        RuntimeError: If there are issues with the molecular conversion process.

    Example:
        >>> conformers = smiles_to_ensemble("CCO", 5)
        >>> print(len(conformers))
        5
        >>> print(conformers[0][:20])
        9
        Conformer 1 from SMI...
    """
    if not smiles_string or not smiles_string.strip():
        raise ValueError("SMILES string cannot be empty or None")

    if not isinstance(num_conformers, int) or num_conformers <= 0:
        raise ValueError("num_conformers must be a positive integer")

    try:
        # Use mol_utils function to generate conformer ensemble
        conformer_data = smiles_to_conformer_ensemble(smiles_string.strip(), num_conformers)

        if not conformer_data:
            raise ValueError("Failed to generate conformers from SMILES")

        xyz_conformers = []

        if not return_full_xyz_str:
            return conformer_data

        for i, (atoms, coordinates) in enumerate(conformer_data):
            if not atoms or not coordinates:
                continue

            if len(atoms) != len(coordinates):
                raise ValueError(f"Mismatch between atoms and coordinates in conformer {i+1}")

            # Format each conformer as XYZ string
            xyz_lines = [str(len(atoms))]
            xyz_lines.append(f"Conformer {i+1} from SMILES: {smiles_string}")

            for atom, coord in zip(atoms, coordinates):
                if len(coord) != 3:
                    raise ValueError(f"Invalid coordinate format in conformer {i+1}")
                xyz_lines.append(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

            xyz_conformers.append("\n".join(xyz_lines))

        if not xyz_conformers:
            raise ValueError("No valid conformers were generated")

        return xyz_conformers

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise RuntimeError(f"Error generating conformer ensemble: {str(e)}")


def save_xyz_file(symbols: List[str], coordinates: List[List[float]], energy: Optional[float], filepath: str, comment: str = "") -> None:
    """Save molecular structure to XYZ file format.

    Args:
        symbols: List of atomic symbols (e.g., ['C', 'H', 'H', 'H']).
        coordinates: List of atomic coordinates [[x1,y1,z1], [x2,y2,z2], ...].
        energy: Energy of the structure in eV, or None if not available.
        filepath: Output file path for the XYZ file.
        comment: Optional comment line for the XYZ file.

    Raises:
        ValueError: If symbols and coordinates lengths don't match or are empty.
        OSError: If file cannot be written to specified path.

    Note:
        Creates parent directories if they don't exist.
    """
    if not any(symbols) or not any(coordinates):
        raise ValueError("Cannot save empty structure")

    if len(symbols) != len(coordinates):
        raise ValueError(f"Number of symbols ({len(symbols)}) must match number of coordinates ({len(coordinates)})")

    # Validate coordinates format
    for i, coord in enumerate(coordinates):
        if len(coord) != 3:
            raise ValueError(f"Coordinate {i} must have exactly 3 components (x, y, z)")

    # Create parent directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    try:
        with open(filepath, 'w') as f:
            f.write(f"{len(symbols)}\n")
            # Format energy in comment line if available
            if energy is not None:
                f.write(f"{comment} | Energy: {energy:.6f} eV\n")
            else:
                f.write(f"{comment}\n")

            for symbol, coord in zip(symbols, coordinates):
                f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    except OSError as e:
        raise OSError(f"Cannot write to file {filepath}: {e}") from e


def save_multi_xyz(structures: List[Tuple[List[str], List[List[float]], float]], filepath: str, comments: Optional[List[str]] = None) -> None:
    """Save multiple molecular structures to a single multi-XYZ file.

    Args:
        structures: List of (symbols, coordinates, energy) tuples.
        filepath: Output file path for the multi-XYZ file.
        comments: Optional list of comments, one for each structure.

    Raises:
        ValueError: If structures list is empty or contains invalid data.
        OSError: If file cannot be written to specified path.

    Note:
        If comments list is shorter than structures list, remaining structures
        get generic "Structure N" comments. Creates parent directories if needed.
    """
    if not structures:
        raise ValueError("Cannot save empty structure list")

    # Validate all structures
    for i, structure in enumerate(structures):
        if len(structure) != 3:
            raise ValueError(f"Structure {i} must be a tuple of (symbols, coordinates, energy)")
        symbols, coordinates, energy = structure
        if len(symbols) != len(coordinates):
            raise ValueError(f"Structure {i}: symbols/coordinates length mismatch")

    # Create parent directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    try:
        with open(filepath, 'w') as f:
            for i, (symbols, coordinates, energy) in enumerate(structures):
                comment = comments[i] if comments and i < len(comments) else f"Structure {i+1}"
                f.write(f"{len(symbols)}\n")
                f.write(f"{comment} | Energy: {energy:.6f} eV\n")
                for symbol, coord in zip(symbols, coordinates):
                    f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
    except OSError as e:
        raise OSError(f"Cannot write to file {filepath}: {e}") from e
