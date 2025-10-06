from uma_geometry_optimizer.structure import Structure


def test_structure_defaults_and_energy_setter():
    s = Structure(symbols=["H", "O", "H"], coordinates=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)], charge=0, multiplicity=1)
    assert s.n_atoms == 3
    assert s.charge == 0 and s.multiplicity == 1
    assert s.energy is None
    s.with_energy(-1.23)
    assert s.energy == -1.23
