import sys
import os
import types
import builtins
import pytest

# Ensure src on sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _make_fake_ase_module():
    # Minimal ASE API used by optimizer
    class Atoms:
        def __init__(self, symbols=None, positions=None):
            self._symbols = list(symbols or [])
            # deep copy positions to avoid accidental aliasing
            self._positions = [list(p) for p in (positions or [])]
            self.calc = None

        def get_positions(self):
            return [list(p) for p in self._positions]

        def set_positions(self, pos):
            self._positions = [list(p) for p in pos]

        def get_potential_energy(self):
            # Return deterministic dummy energy
            return 42.0

        def get_chemical_symbols(self):
            return list(self._symbols)

    class BFGS:
        def __init__(self, atoms, logfile=None):
            self.atoms = atoms
            self._steps = 1
        def run(self):
            # No-op optimization
            self._steps = 2
        def get_number_of_steps(self):
            return self._steps

    ase = types.ModuleType("ase")
    ase.Atoms = Atoms
    optimize = types.ModuleType("ase.optimize")
    optimize.BFGS = BFGS
    ase.optimize = optimize

    # Also add ase.data.chemical_symbols for mol_utils helpers
    data = types.ModuleType("ase.data")
    chem = [None] * 5
    chem += ["C"]  # index 6
    data.chemical_symbols = ["?", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
    ase.data = data
    return ase


def _make_fake_torch_sim_module():
    mod = types.ModuleType("torch_sim")
    def generate_energy_convergence_fn(*a, **k):
        return lambda *a, **k: True
    mod.generate_energy_convergence_fn = generate_energy_convergence_fn

    # Minimal submodules referenced in batch path
    models = types.ModuleType("torch_sim.models")
    atomistic = types.ModuleType("torch_sim.models.fairchem")
    class FairChemModel:
        def __init__(self, model=None, model_name=None, cpu=True, task_name="omol"):
            pass
    atomistic.FairChemModel = FairChemModel
    mod.models = models
    sys.modules["torch_sim.models"] = models
    sys.modules["torch_sim.models.fairchem"] = atomistic

    optimizers = types.ModuleType("torch_sim.optimizers")
    optimizers.gradient_descent = lambda *a, **k: None
    optimizers.fire = lambda *a, **k: None
    sys.modules["torch_sim.optimizers"] = optimizers

    autobatching = types.ModuleType("torch_sim.autobatching")
    class InFlightAutoBatcher:
        def __init__(self, *a, **k):
            pass
    autobatching.InFlightAutoBatcher = InFlightAutoBatcher
    sys.modules["torch_sim.autobatching"] = autobatching

    runners = types.ModuleType("torch_sim.runners")
    class _State:
        def __init__(self):
            self.energy = [types.SimpleNamespace(item=lambda: 0.0)]
        def to_atoms(self):
            ase = sys.modules.get("ase") or _make_fake_ase_module()
            return [ase.Atoms(symbols=["H"], positions=[[0.0,0.0,0.0]])]
    def optimize(system, model, optimizer, convergence_fn, autobatcher, steps_between_swaps):
        return _State()
    runners.optimize = optimize
    sys.modules["torch_sim.runners"] = runners

    # IO helper
    io = types.ModuleType("torch_sim.io")
    def atoms_to_state(ase_structures, device=None, dtype=None):
        return object()
    io.atoms_to_state = atoms_to_state
    sys.modules["torch_sim.io"] = io

    return mod


# Pre-install fakes so imports during test collection succeed
if "ase" not in sys.modules:
    sys.modules["ase"] = _make_fake_ase_module()
    sys.modules["ase.optimize"] = sys.modules["ase"].optimize
    sys.modules["ase.data"] = sys.modules["ase"].data
if "torch_sim" not in sys.modules:
    sys.modules["torch_sim"] = _make_fake_torch_sim_module()
# Do NOT stub torch or numpy; use the real installed libraries


@pytest.fixture(autouse=True)
def fake_heavy_modules(monkeypatch):
    # Keep fakes present for each test
    yield
    # No cleanup
