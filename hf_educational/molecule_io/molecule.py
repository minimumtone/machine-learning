"""
Molecule class for storing atomic coordinates and properties.
"""

import numpy as np
from typing import List, Tuple, Optional

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30
}

ATOMIC_SYMBOLS = {v: k for k, v in ATOMIC_NUMBERS.items()}


class Molecule:
    """
    Represents a molecular system with atoms and their coordinates.
    
    Attributes:
        atoms: List of atomic numbers
        coords: Atomic coordinates in Bohr (N_atoms x 3)
        charge: Total molecular charge
        multiplicity: Spin multiplicity (2S+1)
        units: 'bohr' or 'angstrom'
    """
    
    def __init__(self, 
                 atoms: List[int], 
                 coords: np.ndarray,
                 charge: int = 0,
                 multiplicity: int = 1,
                 units: str = 'angstrom'):
        """
        Initialize molecule.
        
        Args:
            atoms: List of atomic numbers
            coords: Atomic coordinates (N_atoms x 3)
            charge: Total molecular charge
            multiplicity: Spin multiplicity (2S+1)
            units: 'bohr' or 'angstrom'
        """
        self.atoms = np.array(atoms, dtype=int)
        self.coords = np.array(coords, dtype=float)
        self.charge = charge
        self.multiplicity = multiplicity
        
        if units.lower() == 'angstrom':
            self.coords *= ANGSTROM_TO_BOHR
        elif units.lower() != 'bohr':
            raise ValueError(f"Unknown units: {units}")
        
        self.n_atoms = len(self.atoms)
        self.n_electrons = self.nuclear_charge() - charge
        
        if self.n_electrons < 0:
            raise ValueError(f"Invalid charge {charge} for molecule")
        
        n_unpaired = multiplicity - 1
        if (self.n_electrons - n_unpaired) % 2 != 0:
            raise ValueError(f"Incompatible multiplicity {multiplicity} with {self.n_electrons} electrons")
        
        self.n_alpha = (self.n_electrons + n_unpaired) // 2
        self.n_beta = (self.n_electrons - n_unpaired) // 2
        
    def nuclear_charge(self) -> int:
        """Total nuclear charge (sum of atomic numbers)."""
        return int(np.sum(self.atoms))
    
    def nuclear_repulsion(self) -> float:
        """
        Calculate nuclear-nuclear repulsion energy.
        
        Returns:
            E_nn in Hartree
        """
        E_nn = 0.0
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                R_ij = np.linalg.norm(self.coords[i] - self.coords[j])
                E_nn += self.atoms[i] * self.atoms[j] / R_ij
        return E_nn
    
    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass in Bohr."""
        masses = np.array([self.atoms[i] for i in range(self.n_atoms)])
        return np.sum(self.coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
    
    def is_closed_shell(self) -> bool:
        """Check if molecule is closed shell (RHF applicable)."""
        return self.multiplicity == 1
    
    def __str__(self) -> str:
        """String representation of molecule."""
        lines = [f"Molecule: {self.n_atoms} atoms, {self.n_electrons} electrons"]
        lines.append(f"Charge: {self.charge}, Multiplicity: {self.multiplicity}")
        lines.append(f"Alpha electrons: {self.n_alpha}, Beta electrons: {self.n_beta}")
        lines.append("\nGeometry (Angstrom):")
        lines.append("  Atom       X           Y           Z")
        for i in range(self.n_atoms):
            symbol = ATOMIC_SYMBOLS.get(self.atoms[i], f"Z{self.atoms[i]}")
            x, y, z = self.coords[i] * BOHR_TO_ANGSTROM
            lines.append(f"  {symbol:4s}  {x:10.6f}  {y:10.6f}  {z:10.6f}")
        return "\n".join(lines)
    
    @classmethod
    def from_xyz_string(cls, xyz_string: str, charge: int = 0, multiplicity: int = 1):
        """
        Create molecule from XYZ format string.
        
        Args:
            xyz_string: XYZ format string
            charge: Molecular charge
            multiplicity: Spin multiplicity
        """
        lines = xyz_string.strip().split('\n')
        n_atoms = int(lines[0])
        
        atoms = []
        coords = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            symbol = parts[0]
            if symbol in ATOMIC_NUMBERS:
                atoms.append(ATOMIC_NUMBERS[symbol])
            else:
                atoms.append(int(symbol))
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        return cls(atoms, np.array(coords), charge, multiplicity, units='angstrom')
    
    @classmethod
    def from_xyz_file(cls, filename: str, charge: int = 0, multiplicity: int = 1):
        """Load molecule from XYZ file."""
        with open(filename, 'r') as f:
            return cls.from_xyz_string(f.read(), charge, multiplicity)
