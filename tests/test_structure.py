import pytest
from uma_geometry_optimizer.structure import Structure


def test_structure_basic_properties_and_methods():
    s = Structure(symbols=["H", "O", "H"], coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert s.n_atoms == 3
    assert s.energy is None

    s2 = s.with_energy(-1.23)
    assert s2 is s
    assert s.energy == -1.23

