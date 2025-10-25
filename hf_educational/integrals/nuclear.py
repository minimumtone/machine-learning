"""
Nuclear attraction integral evaluation for Gaussian-type orbitals.

Implements analytical evaluation of nuclear attraction integrals:
V_μν = ∫ χ_μ(r) (-Z_C/|r-C|) χ_ν(r) dr

Using Gaussian product theorem and Boys function.
"""

import numpy as np
from typing import TYPE_CHECKING
from .boys import boys_function_array

if TYPE_CHECKING:
    from basis.basis_set import PrimitiveGTO, ContractedGTO, BasisSet
    from molecule_io.molecule import Molecule


def nuclear_attraction_primitive(prim1: 'PrimitiveGTO', prim2: 'PrimitiveGTO',
                                 C: np.ndarray, Z_C: int) -> float:
    """
    Compute nuclear attraction integral between two primitive GTOs.
    
    V = -Z_C ∫ χ_a(r) |r-C|^(-1) χ_b(r) dr
    
    Uses Gaussian product theorem and Boys function.
    
    Args:
        prim1, prim2: Primitive GTOs
        C: Nuclear center coordinates
        Z_C: Nuclear charge
    
    Returns:
        Nuclear attraction integral value
    """
    alpha = prim1.alpha
    beta = prim2.alpha
    
    A = prim1.center
    B = prim2.center
    
    la, ma, na = prim1.l, prim1.m, prim1.n
    lb, mb, nb = prim2.l, prim2.m, prim2.n
    
    p = alpha + beta
    mu = alpha * beta / p
    
    AB = A - B
    AB2 = np.dot(AB, AB)
    
    P = (alpha * A + beta * B) / p
    
    K_AB = np.exp(-mu * AB2)
    
    PC = P - C
    PC2 = np.dot(PC, PC)
    
    t = p * PC2
    
    L_max = la + lb + ma + mb + na + nb
    
    F = boys_function_array(L_max, t)
    
    V = nuclear_attraction_recursive(la, lb, ma, mb, na, nb,
                                     P - A, P - B, P - C,
                                     p, F)
    
    prefactor = -2.0 * np.pi / p * K_AB * Z_C
    
    return prefactor * V


def nuclear_attraction_recursive(la: int, lb: int, ma: int, mb: int, na: int, nb: int,
                                 PA: np.ndarray, PB: np.ndarray, PC: np.ndarray,
                                 p: float, F: np.ndarray) -> float:
    """
    Recursive evaluation of nuclear attraction auxiliary integrals.
    
    Uses Obara-Saika recursion relations for nuclear attraction.
    
    Args:
        la, lb, ma, mb, na, nb: Angular momentum quantum numbers
        PA, PB, PC: P-A, P-B, P-C vectors
        p: Combined exponent
        F: Array of Boys function values
    
    Returns:
        Auxiliary integral value
    """
    L = la + lb + ma + mb + na + nb
    
    if L == 0:
        return F[0]
    
    if la > 0:
        term1 = PA[0] * nuclear_attraction_recursive(la - 1, lb, ma, mb, na, nb,
                                                     PA, PB, PC, p, F[:-1])
        
        term2 = -PC[0] * nuclear_attraction_recursive(la - 1, lb, ma, mb, na, nb,
                                                      PA, PB, PC, p, F[1:])
        
        result = term1 + term2
        
        if la > 1:
            coeff = (la - 1) / (2.0 * p)
            result += coeff * (nuclear_attraction_recursive(la - 2, lb, ma, mb, na, nb,
                                                           PA, PB, PC, p, F[:-1]) -
                              nuclear_attraction_recursive(la - 2, lb, ma, mb, na, nb,
                                                           PA, PB, PC, p, F[1:]))
        
        if lb > 0:
            coeff = lb / (2.0 * p)
            result += coeff * (nuclear_attraction_recursive(la - 1, lb - 1, ma, mb, na, nb,
                                                           PA, PB, PC, p, F[:-1]) -
                              nuclear_attraction_recursive(la - 1, lb - 1, ma, mb, na, nb,
                                                           PA, PB, PC, p, F[1:]))
        
        return result
    
    if lb > 0:
        term1 = PB[0] * nuclear_attraction_recursive(la, lb - 1, ma, mb, na, nb,
                                                     PA, PB, PC, p, F[:-1])
        
        term2 = -PC[0] * nuclear_attraction_recursive(la, lb - 1, ma, mb, na, nb,
                                                      PA, PB, PC, p, F[1:])
        
        result = term1 + term2
        
        if lb > 1:
            coeff = (lb - 1) / (2.0 * p)
            result += coeff * (nuclear_attraction_recursive(la, lb - 2, ma, mb, na, nb,
                                                           PA, PB, PC, p, F[:-1]) -
                              nuclear_attraction_recursive(la, lb - 2, ma, mb, na, nb,
                                                           PA, PB, PC, p, F[1:]))
        
        return result
    
    if ma > 0:
        term1 = PA[1] * nuclear_attraction_recursive(la, lb, ma - 1, mb, na, nb,
                                                     PA, PB, PC, p, F[:-1])
        
        term2 = -PC[1] * nuclear_attraction_recursive(la, lb, ma - 1, mb, na, nb,
                                                      PA, PB, PC, p, F[1:])
        
        result = term1 + term2
        
        if ma > 1:
            coeff = (ma - 1) / (2.0 * p)
            result += coeff * (nuclear_attraction_recursive(la, lb, ma - 2, mb, na, nb,
                                                           PA, PB, PC, p, F[:-1]) -
                              nuclear_attraction_recursive(la, lb, ma - 2, mb, na, nb,
                                                           PA, PB, PC, p, F[1:]))
        
        if mb > 0:
            coeff = mb / (2.0 * p)
            result += coeff * (nuclear_attraction_recursive(la, lb, ma - 1, mb - 1, na, nb,
                                                           PA, PB, PC, p, F[:-1]) -
                              nuclear_attraction_recursive(la, lb, ma - 1, mb - 1, na, nb,
                                                           PA, PB, PC, p, F[1:]))
        
        return result
    
    if mb > 0:
        term1 = PB[1] * nuclear_attraction_recursive(la, lb, ma, mb - 1, na, nb,
                                                     PA, PB, PC, p, F[:-1])
        
        term2 = -PC[1] * nuclear_attraction_recursive(la, lb, ma, mb - 1, na, nb,
                                                      PA, PB, PC, p, F[1:])
        
        result = term1 + term2
        
        if mb > 1:
            coeff = (mb - 1) / (2.0 * p)
            result += coeff * (nuclear_attraction_recursive(la, lb, ma, mb - 2, na, nb,
                                                           PA, PB, PC, p, F[:-1]) -
                              nuclear_attraction_recursive(la, lb, ma, mb - 2, na, nb,
                                                           PA, PB, PC, p, F[1:]))
        
        return result
    
    if na > 0:
        term1 = PA[2] * nuclear_attraction_recursive(la, lb, ma, mb, na - 1, nb,
                                                     PA, PB, PC, p, F[:-1])
        
        term2 = -PC[2] * nuclear_attraction_recursive(la, lb, ma, mb, na - 1, nb,
                                                      PA, PB, PC, p, F[1:])
        
        result = term1 + term2
        
        if na > 1:
            coeff = (na - 1) / (2.0 * p)
            result += coeff * (nuclear_attraction_recursive(la, lb, ma, mb, na - 2, nb,
                                                           PA, PB, PC, p, F[:-1]) -
                              nuclear_attraction_recursive(la, lb, ma, mb, na - 2, nb,
                                                           PA, PB, PC, p, F[1:]))
        
        if nb > 0:
            coeff = nb / (2.0 * p)
            result += coeff * (nuclear_attraction_recursive(la, lb, ma, mb, na - 1, nb - 1,
                                                           PA, PB, PC, p, F[:-1]) -
                              nuclear_attraction_recursive(la, lb, ma, mb, na - 1, nb - 1,
                                                           PA, PB, PC, p, F[1:]))
        
        return result
    
    if nb > 0:
        term1 = PB[2] * nuclear_attraction_recursive(la, lb, ma, mb, na, nb - 1,
                                                     PA, PB, PC, p, F[:-1])
        
        term2 = -PC[2] * nuclear_attraction_recursive(la, lb, ma, mb, na, nb - 1,
                                                      PA, PB, PC, p, F[1:])
        
        result = term1 + term2
        
        if nb > 1:
            coeff = (nb - 1) / (2.0 * p)
            result += coeff * (nuclear_attraction_recursive(la, lb, ma, mb, na, nb - 2,
                                                           PA, PB, PC, p, F[:-1]) -
                              nuclear_attraction_recursive(la, lb, ma, mb, na, nb - 2,
                                                           PA, PB, PC, p, F[1:]))
        
        return result
    
    return F[0]


def nuclear_attraction_contracted(cgto1: 'ContractedGTO', cgto2: 'ContractedGTO',
                                  C: np.ndarray, Z_C: int) -> float:
    """
    Compute nuclear attraction integral between two contracted GTOs.
    
    V = sum_p sum_q d_p d_q V_pq
    
    Args:
        cgto1, cgto2: Contracted GTOs
        C: Nuclear center
        Z_C: Nuclear charge
    
    Returns:
        Nuclear attraction integral value
    """
    V = 0.0
    for prim1 in cgto1.primitives:
        for prim2 in cgto2.primitives:
            V += prim1.coeff * prim2.coeff * nuclear_attraction_primitive(prim1, prim2, C, Z_C)
    return V


def compute_nuclear_attraction_matrix(basis: 'BasisSet', molecule: 'Molecule') -> np.ndarray:
    """
    Compute nuclear attraction matrix V for entire basis set.
    
    V_μν = sum_C (-Z_C) ∫ χ_μ(r) |r-C|^(-1) χ_ν(r) dr
    
    Args:
        basis: BasisSet object
        molecule: Molecule object
    
    Returns:
        Nuclear attraction matrix (n_basis x n_basis)
    """
    n = basis.n_basis
    V = np.zeros((n, n))
    
    for atom_idx in range(molecule.n_atoms):
        C = molecule.coords[atom_idx]
        Z_C = molecule.atoms[atom_idx]
        
        for i in range(n):
            for j in range(i + 1):
                V_ij = nuclear_attraction_contracted(basis[i].cgto, basis[j].cgto, C, Z_C)
                V[i, j] += V_ij
                if i != j:
                    V[j, i] += V_ij
    
    return V
