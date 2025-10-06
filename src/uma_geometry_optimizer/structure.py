from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Structure:
    """Container for a molecular structure.

    Attributes:
        symbols: List of atomic symbols.
        coordinates: Nx3 list of floats for atomic positions.
        energy: Optional energy value (eV) if available.
        comment: Optional comment/metadata string.
    """
    symbols: List[str]
    coordinates: List[List[float]]
    energy: Optional[float] = None
    comment: str = ""
    charge: int = 0
    multiplicity: int = 1

    # room for future metadata without breaking API
    metadata: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    def with_energy(self, energy: Optional[float]) -> "Structure":
        self.energy = energy
        return self

