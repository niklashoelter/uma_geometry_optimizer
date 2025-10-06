import uma_geometry_optimizer as pkg
from uma_geometry_optimizer.structure import Structure
from uma_geometry_optimizer.config import Config


def test_optimize_single_smiles_monkeypatched(monkeypatch, tmp_path):
    calls = {}

    def fake_smiles_to_xyz(smiles):
        return Structure(["H"], [[0.0, 0.0, 0.0]])

    def fake_optimize_single_structure(s, cfg=None, calculator=None):
        s.energy = -1.0
        return s

    def fake_save_xyz(s, path):
        calls["saved"] = path

    monkeypatch.setattr(pkg, "smiles_to_xyz", fake_smiles_to_xyz)
    monkeypatch.setattr(pkg, "optimize_single_structure", fake_optimize_single_structure)
    monkeypatch.setattr(pkg, "save_xyz_file", fake_save_xyz)

    out = tmp_path / "a.xyz"
    s = pkg.optimize_single_smiles("H", output_file=str(out), config=Config())
    assert isinstance(s, Structure)
    assert s.energy == -1.0
    assert calls["saved"] == str(out)


def test_optimize_single_xyz_file(monkeypatch, tmp_path):
    calls = {}

    def fake_read_xyz(p):
        return Structure(["H"], [[0.0, 0.0, 0.0]])

    def fake_optimize_single_structure(s, cfg=None, calculator=None):
        s.energy = -2.0
        return s

    def fake_save_xyz(s, path):
        calls["saved"] = path

    monkeypatch.setattr(pkg, "read_xyz", fake_read_xyz)
    monkeypatch.setattr(pkg, "optimize_single_structure", fake_optimize_single_structure)
    monkeypatch.setattr(pkg, "save_xyz_file", fake_save_xyz)

    out = tmp_path / "b.xyz"
    s = pkg.optimize_single_xyz_file(str(tmp_path / "in.xyz"), output_file=str(out), config=Config())
    assert isinstance(s, Structure)
    assert s.energy == -2.0
    assert calls["saved"] == str(out)


def test_optimize_smiles_ensemble(monkeypatch, tmp_path):
    calls = {}
    structures = [Structure(["H"], [[0.0, 0.0, 0.0]]), Structure(["He"], [[1.0, 0.0, 0.0]])]

    def fake_smiles_to_ensemble(smiles, num):
        return structures

    def fake_optimize_batch(structs, cfg=None):
        for i, s in enumerate(structs):
            s.energy = float(-i)
        return structs

    def fake_save_multi(structs, path, comments=None):
        calls["saved"] = path
        calls["comments"] = list(comments or [])

    monkeypatch.setattr(pkg, "smiles_to_ensemble", fake_smiles_to_ensemble)
    monkeypatch.setattr(pkg, "optimize_structure_batch", fake_optimize_batch)
    monkeypatch.setattr(pkg, "save_multi_xyz", fake_save_multi)

    out = tmp_path / "c.xyz"
    res = pkg.optimize_smiles_ensemble("H", num_conformers=2, output_file=str(out), config=Config())
    assert isinstance(res, list) and len(res) == 2
    assert calls["saved"] == str(out)
    assert all("Optimized conformer" in c for c in calls["comments"])
