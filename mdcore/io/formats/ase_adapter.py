"""ASE (Atomic Simulation Environment) adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ...system import MDState

# Optional ASE import
try:
    from ase import Atoms
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    HAS_ASE = True
except ImportError:
    HAS_ASE = False


def check_ase() -> None:
    """Check if ASE is available."""
    if not HAS_ASE:
        raise ImportError("ASE required: pip install ase")


class ASEAdapter:
    """
    Adapter for converting between MDState and ASE Atoms objects.

    ASE provides a rich ecosystem for atomic simulations, including:
    - Many file format readers/writers
    - Calculator interfaces
    - Visualization tools

    Example:
        adapter = ASEAdapter()

        # Convert MDState to ASE Atoms
        atoms = adapter.to_ase(state)

        # Use ASE functionality
        ase_write("structure.cif", atoms)

        # Convert back
        state = adapter.from_ase(atoms)
    """

    def __init__(self, elements: list[str] | None = None) -> None:
        """
        Initialize ASE adapter.

        Args:
            elements: Default element symbols if not in state.
        """
        check_ase()
        self.default_elements = elements

    def to_ase(
        self,
        state: MDState,
        elements: list[str] | None = None,
    ) -> Atoms:
        """
        Convert MDState to ASE Atoms object.

        Args:
            state: MD state to convert.
            elements: Element symbols (overrides defaults).

        Returns:
            ASE Atoms object.
        """
        positions = state.positions

        # Get elements
        if elements is not None:
            symbols = elements
        elif hasattr(state, "elements") and state.elements is not None:
            symbols = state.elements
        elif self.default_elements is not None:
            symbols = self.default_elements
        else:
            symbols = ["X"] * state.n_atoms

        # Create Atoms object
        atoms = Atoms(symbols=symbols, positions=positions)

        # Set cell if available
        if state.box is not None:
            atoms.set_cell(state.box.vectors)
            atoms.set_pbc(True)

        # Set velocities if available
        if state.velocities is not None:
            atoms.set_velocities(state.velocities)

        # Set masses
        atoms.set_masses(state.masses)

        return atoms

    def from_ase(
        self,
        atoms: Atoms,
        forces: NDArray[np.floating] | None = None,
    ) -> dict[str, Any]:
        """
        Convert ASE Atoms to state dictionary.

        Args:
            atoms: ASE Atoms object.
            forces: Optional forces array.

        Returns:
            Dictionary suitable for creating MDState.
        """
        result = {
            "positions": atoms.get_positions(),
            "masses": atoms.get_masses(),
            "elements": atoms.get_chemical_symbols(),
            "n_atoms": len(atoms),
        }

        # Velocities
        if atoms.get_velocities() is not None:
            result["velocities"] = atoms.get_velocities()

        # Box/cell
        if atoms.get_pbc().any():
            result["box_vectors"] = atoms.get_cell()[:]

        # Forces
        if forces is not None:
            result["forces"] = forces
        elif atoms.calc is not None:
            import contextlib

            with contextlib.suppress(Exception):
                result["forces"] = atoms.get_forces()

        return result


def read_ase(filename: str, index: int | str = -1, **kwargs) -> list[dict]:
    """
    Read structure(s) using ASE's universal reader.

    ASE supports many formats: xyz, pdb, cif, vasp, extxyz, etc.

    Args:
        filename: Input file path.
        index: Frame index or slice string (e.g., ":", "-1", "0:10").
        **kwargs: Additional arguments passed to ase.io.read.

    Returns:
        List of state dictionaries.
    """
    check_ase()

    atoms_list = ase_read(filename, index=index, **kwargs)

    # Ensure it's a list
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    adapter = ASEAdapter()
    return [adapter.from_ase(atoms) for atoms in atoms_list]


def write_ase(
    filename: str,
    states: list[dict] | dict,
    elements: list[str] | None = None,
    **kwargs,
) -> None:
    """
    Write structure(s) using ASE's universal writer.

    Args:
        filename: Output file path.
        states: State dictionary or list of state dictionaries.
        elements: Element symbols.
        **kwargs: Additional arguments passed to ase.io.write.
    """
    check_ase()

    if not isinstance(states, list):
        states = [states]

    atoms_list = []
    for state in states:
        # Create minimal Atoms from dict
        positions = state["positions"]
        symbols = elements or state.get("elements", ["X"] * len(positions))

        atoms = Atoms(symbols=symbols, positions=positions)

        if "box_vectors" in state and state["box_vectors"] is not None:
            atoms.set_cell(state["box_vectors"])
            atoms.set_pbc(True)

        if "velocities" in state and state["velocities"] is not None:
            atoms.set_velocities(state["velocities"])

        if "masses" in state:
            atoms.set_masses(state["masses"])

        atoms_list.append(atoms)

    if len(atoms_list) == 1:
        ase_write(filename, atoms_list[0], **kwargs)
    else:
        ase_write(filename, atoms_list, **kwargs)
