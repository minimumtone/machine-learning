"""
LCAO-Hartree-Fock Educational Program

A comprehensive implementation of Restricted and Unrestricted Hartree-Fock
methods for educational purposes. This package provides:

- RHF/UHF SCF calculations
- Gaussian basis set support (STO-3G, 6-31G, etc.)
- Full integral evaluation from scratch
- DIIS convergence acceleration
- Mulliken population analysis
- Interactive visualization of MO energies, densities, and convergence
- Educational UI with J/K contribution sliders

Theory implementation follows standard quantum chemistry textbooks
(Szabo & Ostlund, Modern Quantum Chemistry).
"""

__version__ = "1.0.0"
__author__ = "Educational HF Project"

from .molecule_io.molecule import Molecule
from .scf.rhf import RHF
from .scf.uhf import UHF

__all__ = ['Molecule', 'RHF', 'UHF']
