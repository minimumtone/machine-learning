"""
Basis set classes for Gaussian-type orbitals (GTOs).

Implements primitive and contracted GTOs with proper normalization.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PrimitiveGTO:
    """
    Primitive Gaussian-type orbital: g(r) = N * x^l * y^m * z^n * exp(-alpha * r^2)
    
    Attributes:
        alpha: Gaussian exponent
        coeff: Contraction coefficient
        l, m, n: Cartesian angular momentum quantum numbers
        center: Atomic center coordinates (3D)
    """
    alpha: float
    coeff: float
    l: int
    m: int
    n: int
    center: np.ndarray
    
    def angular_momentum(self) -> int:
        """Total angular momentum L = l + m + n."""
        return self.l + self.m + self.n
    
    def normalization_constant(self) -> float:
        """
        Analytical normalization constant for primitive GTO.
        
        N = (2*alpha/pi)^(3/4) * (4*alpha)^(L/2) / sqrt((2l-1)!! * (2m-1)!! * (2n-1)!!)
        """
        from math import pi, sqrt
        
        def double_factorial(n):
            if n <= 0:
                return 1
            result = 1
            for i in range(n, 0, -2):
                result *= i
            return result
        
        L = self.l + self.m + self.n
        
        prefactor = (2.0 * self.alpha / pi) ** 0.75
        prefactor *= (4.0 * self.alpha) ** (L / 2.0)
        
        denom = sqrt(double_factorial(2*self.l - 1) * 
                     double_factorial(2*self.m - 1) * 
                     double_factorial(2*self.n - 1))
        
        return prefactor / denom


@dataclass
class ContractedGTO:
    """
    Contracted Gaussian-type orbital: sum of primitive GTOs.
    
    chi(r) = sum_p d_p * g_p(r)
    
    Attributes:
        primitives: List of primitive GTOs
        l, m, n: Cartesian angular momentum
        center: Atomic center
        atom_idx: Index of atom this basis function belongs to
    """
    primitives: List[PrimitiveGTO]
    l: int
    m: int
    n: int
    center: np.ndarray
    atom_idx: int
    
    def angular_momentum(self) -> int:
        """Total angular momentum L = l + m + n."""
        return self.l + self.m + self.n
    
    def normalize(self):
        """
        Normalize the contracted GTO.
        
        Ensures <chi|chi> = 1 by computing the self-overlap and rescaling.
        """
        from integrals.overlap import overlap_primitive
        
        norm_sq = 0.0
        for p1 in self.primitives:
            for p2 in self.primitives:
                S = overlap_primitive(p1, p2)
                norm_sq += p1.coeff * p2.coeff * S
        
        norm = np.sqrt(norm_sq)
        if norm > 1e-10:
            for p in self.primitives:
                p.coeff /= norm
    
    def shell_type(self) -> str:
        """Return shell type label (s, p, d, f, ...)."""
        L = self.angular_momentum()
        labels = ['s', 'p', 'd', 'f', 'g', 'h', 'i']
        return labels[L] if L < len(labels) else f'L{L}'


class BasisFunction:
    """
    Wrapper for a single basis function (AO) in the basis set.
    
    Each basis function is a contracted GTO with specific angular momentum.
    """
    
    def __init__(self, cgto: ContractedGTO, label: str):
        self.cgto = cgto
        self.label = label
        self.l = cgto.l
        self.m = cgto.m
        self.n = cgto.n
        self.center = cgto.center
        self.atom_idx = cgto.atom_idx
    
    def __repr__(self):
        return f"BasisFunction({self.label}, L={self.angular_momentum()})"
    
    def angular_momentum(self) -> int:
        return self.l + self.m + self.n


class BasisSet:
    """
    Complete basis set for a molecule.
    
    Manages all basis functions (AOs) for all atoms.
    """
    
    def __init__(self, molecule, basis_name: str = 'sto-3g'):
        """
        Initialize basis set for a molecule.
        
        Args:
            molecule: Molecule object
            basis_name: Name of basis set (e.g., 'sto-3g', '6-31g')
        """
        self.molecule = molecule
        self.basis_name = basis_name.lower()
        self.basis_functions: List[BasisFunction] = []
        
        self._build_basis()
        self.n_basis = len(self.basis_functions)
    
    def _build_basis(self):
        """Build basis functions for all atoms in molecule."""
        from .basis_parser import get_basis_data
        
        for atom_idx, atomic_num in enumerate(self.molecule.atoms):
            center = self.molecule.coords[atom_idx]
            basis_data = get_basis_data(atomic_num, self.basis_name)
            
            for shell in basis_data:
                shell_type = shell['shell_type']
                exponents = shell['exponents']
                coefficients = shell['coefficients']
                
                cgtos = self._expand_shell(shell_type, exponents, coefficients, 
                                          center, atom_idx)
                
                for cgto in cgtos:
                    cgto.normalize()
                    label = self._make_label(atom_idx, cgto)
                    bf = BasisFunction(cgto, label)
                    self.basis_functions.append(bf)
    
    def _expand_shell(self, shell_type: str, exponents: List[float], 
                     coefficients: List[float], center: np.ndarray, 
                     atom_idx: int) -> List[ContractedGTO]:
        """
        Expand a shell into individual Cartesian GTOs.
        
        For example, a p-shell expands into px, py, pz.
        """
        cgtos = []
        
        angular_momentum_map = {
            's': [(0, 0, 0)],
            'p': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            'd': [(2, 0, 0), (0, 2, 0), (0, 0, 2), 
                  (1, 1, 0), (1, 0, 1), (0, 1, 1)],
            'f': [(3, 0, 0), (0, 3, 0), (0, 0, 3),
                  (2, 1, 0), (2, 0, 1), (1, 2, 0),
                  (0, 2, 1), (1, 0, 2), (0, 1, 2),
                  (1, 1, 1)]
        }
        
        if shell_type not in angular_momentum_map:
            raise ValueError(f"Unsupported shell type: {shell_type}")
        
        for l, m, n in angular_momentum_map[shell_type]:
            primitives = []
            for alpha, coeff in zip(exponents, coefficients):
                norm = self._primitive_normalization(alpha, l, m, n)
                prim = PrimitiveGTO(alpha, coeff * norm, l, m, n, center)
                primitives.append(prim)
            
            cgto = ContractedGTO(primitives, l, m, n, center, atom_idx)
            cgtos.append(cgto)
        
        return cgtos
    
    def _primitive_normalization(self, alpha: float, l: int, m: int, n: int) -> float:
        """Calculate normalization constant for primitive GTO."""
        from math import pi, sqrt
        
        def double_factorial(n):
            if n <= 0:
                return 1
            result = 1
            for i in range(n, 0, -2):
                result *= i
            return result
        
        L = l + m + n
        
        prefactor = (2.0 * alpha / pi) ** 0.75
        prefactor *= (4.0 * alpha) ** (L / 2.0)
        
        denom = sqrt(double_factorial(2*l - 1) * 
                     double_factorial(2*m - 1) * 
                     double_factorial(2*n - 1))
        
        return prefactor / denom
    
    def _make_label(self, atom_idx: int, cgto: ContractedGTO) -> str:
        """Create human-readable label for basis function."""
        from molecule_io.molecule import ATOMIC_SYMBOLS
        
        atom_symbol = ATOMIC_SYMBOLS.get(self.molecule.atoms[atom_idx], 
                                         f"Z{self.molecule.atoms[atom_idx]}")
        
        l, m, n = cgto.l, cgto.m, cgto.n
        L = l + m + n
        
        if L == 0:
            orbital = 's'
        elif L == 1:
            if l == 1:
                orbital = 'px'
            elif m == 1:
                orbital = 'py'
            else:
                orbital = 'pz'
        elif L == 2:
            if l == 2:
                orbital = 'dxx'
            elif m == 2:
                orbital = 'dyy'
            elif n == 2:
                orbital = 'dzz'
            elif l == 1 and m == 1:
                orbital = 'dxy'
            elif l == 1 and n == 1:
                orbital = 'dxz'
            else:
                orbital = 'dyz'
        else:
            orbital = f'L{L}({l},{m},{n})'
        
        return f"{atom_symbol}{atom_idx+1}_{orbital}"
    
    def __len__(self):
        return self.n_basis
    
    def __getitem__(self, idx):
        return self.basis_functions[idx]
    
    def __repr__(self):
        return f"BasisSet({self.basis_name}, {self.n_basis} functions)"
