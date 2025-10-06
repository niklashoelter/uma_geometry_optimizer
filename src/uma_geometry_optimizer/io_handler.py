"""Input/Output handler for molecular structures.

This module provides functions for reading molecular structures from various formats
and converting between different representations.
"""

import os
import glob
from typing import List, Optional, Union, Tuple

from .structure import Structure
from .mol_utils import smiles_to_structure as _smiles_to_structure_util, smiles_to_conformer_ensemble as _smiles_to_ensemble_util


def read_xyz(file_path: str, charge: int = 0, multiplicity: int = 1) -> Structure:
    """Read an XYZ file and return a Structure.

    Args:
        file_path: Path to the XYZ file to read.
        charge: Optional total charge to set on the structure (default: 0).
        multiplicity: Optional spin multiplicity to set (default: 1).

    Returns:
        Structure object with symbols, coordinates, and optional comment/energy.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is invalid.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    symbols: List[str] = []
    coordinates: List[Tuple[float, float, float]] = []
    comment = ""
    energy: Optional[float] = None

    try:
        with open(file_path, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]

        try:
            num_atoms = int(lines[0])
        except Exception:
            raise ValueError("First line must contain the number of atoms as an integer")

        if len(lines) < 2 + num_atoms:
            raise ValueError(f"Expected {num_atoms} atom lines, but found {max(0, len(lines)-2)}")
        comment = lines[1] if len(lines) > 1 else ""

        if "Energy:" in comment:
            try:
                after = comment.split("Energy:", 1)[1].strip()
                val = after.split()[0]
                energy = float(val)
            except Exception:
                energy = None

        for i in range(num_atoms):
            parts = lines[2 + i].split()
            if len(parts) < 4:
                raise ValueError(f"Line {i+3} must contain at least 4 elements: symbol x y z")
            symbol = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                raise ValueError(f"Invalid coordinates in line {i+3}: {parts[1:4]}")
            symbols.append(symbol)
            coordinates.append((x, y, z))

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise ValueError(f"Error reading XYZ file: {str(e)}")

    return Structure(symbols=symbols, coordinates=coordinates, energy=energy, comment=comment, charge=charge, multiplicity=multiplicity)


def read_multi_xyz(file_path: str, charge: int = 0, multiplicity: int = 1) -> List[Structure]:
    """Read an XYZ file containing multiple structures.

    Args:
        file_path: Path to the multi-structure XYZ file.
        charge: Optional total charge to set on all returned structures (default: 0).
        multiplicity: Optional spin multiplicity to set (default: 1).

    Returns:
        List of Structure objects.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is invalid.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    structures: List[Structure] = []

    try:
        with open(file_path, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]

        i = 0
        while i < len(lines):
            if lines[i].strip() == '':
                i += 1
                continue

            try:
                num_atoms = int(lines[i].strip())
            except ValueError:
                i += 1
                continue

            if i + 1 + num_atoms >= len(lines):
                break

            comment = lines[i + 1] if (i + 1) < len(lines) else ""
            energy: Optional[float] = None
            if "Energy:" in comment:
                try:
                    after = comment.split("Energy:", 1)[1].strip()
                    val = after.split()[0]
                    energy = float(val)
                except Exception:
                    energy = None

            symbols: List[str] = []
            coordinates: List[Tuple[float, float, float]] = []

            valid = True
            for j in range(num_atoms):
                parts = lines[i + 2 + j].split()
                if len(parts) < 4:
                    valid = False
                    break
                symbol = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    valid = False
                    break
                symbols.append(symbol)
                coordinates.append((x, y, z))

            if valid and len(symbols) == num_atoms:
                structures.append(Structure(symbols=symbols, coordinates=coordinates, energy=energy, comment=comment, charge=charge, multiplicity=multiplicity))

            i = i + 2 + num_atoms

    except Exception as e:
        raise ValueError(f"Error reading multi-XYZ file: {str(e)}")

    return structures


def read_xyz_directory(directory_path: str, charge: int = 0, multiplicity: int = 1) -> List[Structure]:
    """Read all XYZ files from a directory.

    Args:
        directory_path: Path to directory containing XYZ files.
        charge: Optional total charge to set on all returned structures (default: 0).
        multiplicity: Optional spin multiplicity to set (default: 1).

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

    structures: List[Structure] = []

    for xyz_file in xyz_files:
        try:
            structures.append(read_xyz(xyz_file, charge=charge, multiplicity=multiplicity))
        except Exception as e:
            print(f"Warning: Failed to read {xyz_file}: {e}")

    if not structures:
        raise ValueError("No valid structures could be read from any XYZ files")

    return structures


def smiles_to_xyz(smiles_string: str, return_full_xyz_str: bool = False) -> Union[Structure, str]:
    """Convert a SMILES string to XYZ structure or string.

    Args:
        smiles_string: A valid SMILES string representing the molecular structure.
        return_full_xyz_str: If True, return an XYZ-format string. Otherwise, a Structure.

    Returns:
        Structure or XYZ string depending on return_full_xyz_str.
    """
    if not smiles_string or not smiles_string.strip():
        raise ValueError("SMILES string cannot be empty or None")

    struct = _smiles_to_structure_util(smiles_string.strip())

    if return_full_xyz_str:
        xyz_lines = [str(struct.n_atoms)]
        xyz_lines.append(f"Generated from SMILES: {smiles_string}")
        for atom, coord in zip(struct.symbols, struct.coordinates):
            xyz_lines.append(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
        return "\n".join(xyz_lines)
    else:
        struct.comment = f"Generated from SMILES: {smiles_string}"
        return struct


def smiles_to_ensemble(smiles_string: str, num_conformers: int, return_full_xyz_str: bool = False):
    """Generate a conformational ensemble from a SMILES string.

    Args:
        smiles_string: A valid SMILES string representing the molecular structure.
        num_conformers: The maximum number of conformers to generate.
        return_full_xyz_str: If True, returns a list of XYZ format strings for each conformer.

    Returns:
        A list of XYZ strings or a list of Structure instances.
    """
    if not smiles_string or not smiles_string.strip():
        raise ValueError("SMILES string cannot be empty or None")
    if not isinstance(num_conformers, int) or num_conformers <= 0:
        raise ValueError("num_conformers must be a positive integer")

    conformers: List[Structure] = _smiles_to_ensemble_util(smiles_string.strip(), num_conformers)
    if not return_full_xyz_str:
        for i, s in enumerate(conformers):
            s.comment = f"Conformer {i+1} from SMILES: {smiles_string}"
        return conformers

    xyz_conformers: List[str] = []
    for i, s in enumerate(conformers):
        xyz_lines = [str(s.n_atoms)]
        xyz_lines.append(f"Conformer {i+1} from SMILES: {smiles_string}")
        for atom, coord in zip(s.symbols, s.coordinates):
            xyz_lines.append(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
        xyz_conformers.append("\n".join(xyz_lines))

    return xyz_conformers


def save_xyz_file(structure: Structure, filepath: str) -> None:
    """Save a single Structure to an XYZ file.

    Args:
        structure: Structure to save. Uses structure.energy in comment if set.
        filepath: Output file path for the XYZ file.
    """
    if not structure or structure.n_atoms == 0:
        raise ValueError("Cannot save empty structure")

    for i, coord in enumerate(structure.coordinates):
        if len(coord) != 3:
            raise ValueError(f"Coordinate {i} must have exactly 3 components (x, y, z)")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(f"{structure.n_atoms}\n")
        if structure.energy is not None:
            f.write(f"{structure.comment} | Energy: {structure.energy:.6f} eV\n")
        else:
            f.write(f"{structure.comment}\n")
        for symbol, coord in zip(structure.symbols, structure.coordinates):
            f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


def save_multi_xyz(structures: List[Structure], filepath: str, comments: Optional[List[str]] = None) -> None:
    """Save multiple structures to a single multi-XYZ file.

    Args:
        structures: List of Structure objects. Energy is optional.
        filepath: Output file path for the multi-XYZ file.
        comments: Optional list of comments, one for each structure.
    """
    if not structures:
        raise ValueError("Cannot save empty structure list")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'w') as f:
        for i, s in enumerate(structures):
            if s.n_atoms != len(s.coordinates):
                raise ValueError(f"Structure {i}: symbols/coordinates length mismatch")
            comment = comments[i] if comments and i < len(comments) else (s.comment or f"Structure {i+1}")
            f.write(f"{s.n_atoms}\n")
            if s.energy is not None:
                f.write(f"{comment} | Energy: {s.energy:.6f} eV\n")
            else:
                f.write(f"{comment}\n")
            for symbol, coord in zip(s.symbols, s.coordinates):
                f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
