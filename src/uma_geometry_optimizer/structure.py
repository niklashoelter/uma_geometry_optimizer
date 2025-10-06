from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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
    coordinates: List[Tuple[float, float, float]]
    charge: int
    multiplicity: int
    energy: Optional[float] = None
    comment: str = ""

    # room for future metadata without breaking API
    metadata: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    def with_energy(self, energy: Optional[float]) -> "Structure":
        self.energy = energy
        return self
