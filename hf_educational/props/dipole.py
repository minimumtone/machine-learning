"""
Dipole moment calculation.

Computes electric dipole moment from nuclear and electronic contributions.
"""

import numpy as np
from typing import Dict

from basis.basis_set import BasisSet
from molecule_io.molecule import Molecule


def compute_dipole_integrals(basis: BasisSet) -> tuple:
    """
    Compute dipole moment integrals.
    
    μ_x = ∫ χ_μ(r) x χ_ν(r) dr
    μ_y = ∫ χ_μ(r) y χ_ν(r) dr
    μ_z = ∫ χ_μ(r) z χ_ν(r) dr
    
    Args:
        basis: BasisSet object
    
    Returns:
        (μ_x, μ_y, μ_z) tuple of dipole integral matrices
    """
    n = basis.n_basis
    
    mu_x = np.zeros((n, n))
    mu_y = np.zeros((n, n))
    mu_z = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1):
            cgto1 = basis[i].cgto
            cgto2 = basis[j].cgto
            
            center = 0.5 * (cgto1.center + cgto2.center)
            
            from integrals.overlap import overlap_contracted
            S_ij = overlap_contracted(cgto1, cgto2)
            
            mu_x[i, j] = center[0] * S_ij
            mu_y[i, j] = center[1] * S_ij
            mu_z[i, j] = center[2] * S_ij
            
            mu_x[j, i] = mu_x[i, j]
            mu_y[j, i] = mu_y[i, j]
            mu_z[j, i] = mu_z[i, j]
    
    return mu_x, mu_y, mu_z


def compute_dipole_moment(P: np.ndarray, basis: BasisSet, 
                         molecule: Molecule) -> Dict[str, float]:
    """
    Compute electric dipole moment.
    
    μ = μ_nuc - μ_elec
    
    where μ_nuc = sum_A Z_A R_A
          μ_elec = sum_μν P_μν <μ|r|ν>
    
    Args:
        P: Density matrix
        basis: BasisSet object
        molecule: Molecule object
    
    Returns:
        Dictionary with dipole moment components and magnitude
    """
    mu_nuc = np.zeros(3)
    for i in range(molecule.n_atoms):
        mu_nuc += molecule.atoms[i] * molecule.coords[i]
    
    mu_x, mu_y, mu_z = compute_dipole_integrals(basis)
    
    mu_elec = np.array([
        np.sum(P * mu_x),
        np.sum(P * mu_y),
        np.sum(P * mu_z)
    ])
    
    mu_total = mu_nuc - mu_elec
    
    from molecule_io.molecule import BOHR_TO_ANGSTROM
    DEBYE_CONVERSION = 2.541746473  # e*Bohr to Debye
    
    mu_magnitude = np.linalg.norm(mu_total)
    
    return {
        'mu_x': mu_total[0],
        'mu_y': mu_total[1],
        'mu_z': mu_total[2],
        'mu_magnitude': mu_magnitude,
        'mu_x_debye': mu_total[0] * DEBYE_CONVERSION,
        'mu_y_debye': mu_total[1] * DEBYE_CONVERSION,
        'mu_z_debye': mu_total[2] * DEBYE_CONVERSION,
        'mu_magnitude_debye': mu_magnitude * DEBYE_CONVERSION
    }


def print_dipole_moment(dipole: Dict[str, float]):
    """
    Print dipole moment results.
    
    Args:
        dipole: Results from compute_dipole_moment
    """
    print(f"\n{'='*60}")
    print("Electric Dipole Moment")
    print(f"{'='*60}")
    
    print("\nCartesian components (Debye):")
    print(f"  μ_x = {dipole['mu_x_debye']:12.6f}")
    print(f"  μ_y = {dipole['mu_y_debye']:12.6f}")
    print(f"  μ_z = {dipole['mu_z_debye']:12.6f}")
    
    print("\nMagnitude:")
    print(f"  |μ| = {dipole['mu_magnitude_debye']:12.6f} Debye")
    print(f"  |μ| = {dipole['mu_magnitude']:12.6f} a.u.")
