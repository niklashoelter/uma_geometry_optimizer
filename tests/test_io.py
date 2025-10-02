from pathlib import Path

import pytest

from uma_geometry_optimizer.io_handler import (
    read_xyz,
    read_multi_xyz,
    read_xyz_directory,
    save_xyz_file,
    save_multi_xyz,
)


def test_save_and_read_xyz_roundtrip(tmp_path: Path):
    symbols = ["C", "H", "H", "H", "H"]
    coords = [
        [0.000000, 0.000000, 0.000000],
        [0.629118, 0.629118, 0.629118],
        [-0.629118, -0.629118, 0.629118],
        [-0.629118, 0.629118, -0.629118],
        [0.629118, -0.629118, -0.629118],
    ]
    out = tmp_path / "methane.xyz"

    save_xyz_file(symbols, coords, energy=None, filepath=str(out), comment="Methane test")

    assert out.exists()
    r_symbols, r_coords = read_xyz(str(out))
    assert r_symbols == symbols
    assert len(r_coords) == len(coords)
    assert pytest.approx(r_coords[0][0], rel=1e-6) == coords[0][0]


def test_save_and_read_multi_xyz_roundtrip(tmp_path: Path):
    s1 = (["H", "H"], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], -1.0)
    s2 = (["He"], [[0.0, 0.0, 0.0]], -0.1)
    out = tmp_path / "two_structures.xyz"

    save_multi_xyz([s1, s2], str(out), comments=["H2", "He"])

    assert out.exists()
    structures = read_multi_xyz(str(out))
    assert len(structures) == 2
    syms1, coords1 = structures[0]
    assert syms1 == ["H", "H"]
    assert len(coords1) == 2


def test_read_example_xyz_files():
    root = Path(__file__).resolve().parents[1]
    example_dir = root / "examples" / "read_multiple_xyz_dir"
    example_file = example_dir / "conf1_sp_geometry.xyz"

    assert example_file.exists(), "Example XYZ file missing"

    syms, coords = read_xyz(str(example_file))
    assert len(syms) == len(coords) > 0

    # Directory read
    structures = read_xyz_directory(str(example_dir))
    assert len(structures) >= 1

