import pytest

from uma_geometry_optimizer.optimizer import optimize_structure_batch
from uma_geometry_optimizer.structure import Structure


def test_optimize_structure_batch_empty_returns_empty():
    assert optimize_structure_batch([]) == []


def test_optimize_structure_batch_raises_on_mismatch():
    s = Structure(["H", "H"], [(0.0,0.0,0.0)], charge=0, multiplicity=1)
    with pytest.raises(ValueError):
        optimize_structure_batch([s])


def test_optimize_structure_batch_raises_on_empty_structure():
    s = Structure([], [], charge=0, multiplicity=1)
    with pytest.raises(ValueError):
        optimize_structure_batch([s])

