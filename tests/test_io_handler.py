from uma_geometry_optimizer.io_handler import read_xyz, read_multi_xyz, save_xyz_file, save_multi_xyz
from uma_geometry_optimizer.structure import Structure


def test_read_and_save_single_xyz_roundtrip(tmp_path):
    xyz = """3
Water | Energy: -1.000 eV
O 0.000000 0.000000 0.000000
H 0.000000 0.000000 1.000000
H 0.000000 1.000000 0.000000
"""
    src = tmp_path / "w.xyz"
    src.write_text(xyz)

    s = read_xyz(str(src), charge=-1, multiplicity=2)
    assert isinstance(s, Structure)
    assert s.n_atoms == 3
    assert s.energy == -1.0
    assert s.charge == -1 and s.multiplicity == 2

    out = tmp_path / "out.xyz"
    save_xyz_file(s, str(out))
    assert out.exists()


def test_read_multi_xyz(tmp_path):
    content = """2
Mol A | Energy: -2.0 eV
H 0 0 0
H 0 0 1

2
Mol B
He 0 0 0
He 0 0 1
"""
    src = tmp_path / "m.xyz"
    src.write_text(content)
    items = read_multi_xyz(str(src), charge=0, multiplicity=1)
    assert len(items) == 2
    assert items[0].energy == -2.0
    assert items[1].energy is None
    assert all(s.charge == 0 and s.multiplicity == 1 for s in items)


def test_save_multi_xyz(tmp_path):
    structs = [
        Structure(["H"], [(0.0,0.0,0.0)], charge=0, multiplicity=1, energy=-1.0),
        Structure(["He"], [(1.0,0.0,0.0)], charge=0, multiplicity=1, energy=None),
    ]
    out = tmp_path / "multi.xyz"
    save_multi_xyz(structs, str(out), comments=["A", "B"])
    data = out.read_text()
    assert "A | Energy: -1.000000 eV" in data
    assert "B\nHe" in data
