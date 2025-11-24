import uma_geometry_optimizer as pkg


def test_package_exports_and_metadata():
    assert hasattr(pkg, "Structure")
    assert hasattr(pkg, "read_xyz")
    assert hasattr(pkg, "optimize_structure_batch")
    assert hasattr(pkg, "load_model_fairchem")
    assert hasattr(pkg, "load_model_torchsim")


def test_optimize_structure_batch_empty_list_returns_empty():
    assert pkg.optimize_structure_batch([]) == []
