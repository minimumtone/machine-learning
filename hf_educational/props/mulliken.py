"""
Mulliken population analysis.

Partitions electron density among atoms based on overlap matrix.
"""

import numpy as np
from typing import Dict, List

from basis.basis_set import BasisSet
from molecule_io.molecule import Molecule, ATOMIC_SYMBOLS


def mulliken_population_analysis(P: np.ndarray, S: np.ndarray, 
                                 basis: BasisSet, molecule: Molecule) -> Dict:
    """
    Perform Mulliken population analysis.
    
    Mulliken charge on atom A:
    q_A = Z_A - sum_{μ in A} sum_ν P_μν S_νμ
    
    Args:
        P: Density matrix
        S: Overlap matrix
        basis: BasisSet object
        molecule: Molecule object
    
    Returns:
        Dictionary with population analysis results
    """
    n_atoms = molecule.n_atoms
    n_basis = basis.n_basis
    
    PS = P @ S
    
    gross_populations = np.zeros(n_basis)
    for mu in range(n_basis):
        gross_populations[mu] = PS[mu, mu]
    
    atomic_populations = np.zeros(n_atoms)
    for mu in range(n_basis):
        atom_idx = basis[mu].atom_idx
        atomic_populations[atom_idx] += gross_populations[mu]
    
    atomic_charges = np.zeros(n_atoms)
    for atom_idx in range(n_atoms):
        Z = molecule.atoms[atom_idx]
        atomic_charges[atom_idx] = Z - atomic_populations[atom_idx]
    
    bond_orders = np.zeros((n_atoms, n_atoms))
    for mu in range(n_basis):
        atom_i = basis[mu].atom_idx
        for nu in range(n_basis):
            atom_j = basis[nu].atom_idx
            if atom_i != atom_j:
                bond_orders[atom_i, atom_j] += PS[mu, nu] * S[nu, mu]
    
    return {
        'gross_populations': gross_populations,
        'atomic_populations': atomic_populations,
        'atomic_charges': atomic_charges,
        'bond_orders': bond_orders
    }


def print_mulliken_analysis(results: Dict, molecule: Molecule):
    """
    Print Mulliken population analysis results.
    
    Args:
        results: Results from mulliken_population_analysis
        molecule: Molecule object
    """
    print(f"\n{'='*60}")
    print("Mulliken Population Analysis")
    print(f"{'='*60}")
    
    print(f"\n{'Atom':>6} {'Element':>8} {'Population':>12} {'Charge':>12}")
    print("-" * 50)
    
    for i in range(molecule.n_atoms):
        symbol = ATOMIC_SYMBOLS.get(molecule.atoms[i], f"Z{molecule.atoms[i]}")
        pop = results['atomic_populations'][i]
        charge = results['atomic_charges'][i]
        print(f"{i+1:6d} {symbol:>8s} {pop:12.6f} {charge:12.6f}")
    
    total_charge = np.sum(results['atomic_charges'])
    print("-" * 50)
    print(f"{'Total':>14s} {np.sum(results['atomic_populations']):12.6f} {total_charge:12.6f}")
    
    print("\nBond Orders (> 0.1):")
    print(f"{'Atom I':>8} {'Atom J':>8} {'Bond Order':>12}")
    print("-" * 40)
    
    for i in range(molecule.n_atoms):
        for j in range(i + 1, molecule.n_atoms):
            bo = results['bond_orders'][i, j]
            if abs(bo) > 0.1:
                symbol_i = ATOMIC_SYMBOLS.get(molecule.atoms[i], f"Z{molecule.atoms[i]}")
                symbol_j = ATOMIC_SYMBOLS.get(molecule.atoms[j], f"Z{molecule.atoms[j]}")
                print(f"{symbol_i}{i+1:d} {symbol_j}{j+1:d} {bo:12.6f}")
