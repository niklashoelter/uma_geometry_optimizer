import pytest

morfeus = pytest.importorskip("morfeus", reason="morfeus-ml not installed; skipping molecule generation tests")
rdkit = pytest.importorskip("rdkit", reason="rdkit not installed; skipping molecule generation tests")

from uma_geometry_optimizer.mol_utils import smiles_to_structure, smiles_to_conformer_ensemble


def test_smiles_to_structure_simple():
    symbols, coords = smiles_to_structure("CCO")
    assert isinstance(symbols, list) and isinstance(coords, list)
    assert len(symbols) == len(coords) > 0
    assert len(coords[0]) == 3


def test_smiles_to_conformer_ensemble_count():
    confs = smiles_to_conformer_ensemble("c1ccccc1", max_num_confs=3)
    assert 1 <= len(confs) <= 3
    s0, c0 = confs[0]
    assert len(s0) == len(c0) > 0

