"""
Overlap integral evaluation for Gaussian-type orbitals.

Implements analytical evaluation of overlap integrals:
S_μν = ∫ χ_μ(r) χ_ν(r) dr

Using Gaussian product theorem and analytical formulas.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from basis.basis_set import PrimitiveGTO, ContractedGTO, BasisSet


def overlap_primitive(prim1: 'PrimitiveGTO', prim2: 'PrimitiveGTO') -> float:
    """
    Compute overlap integral between two primitive GTOs.
    
    Uses Gaussian product theorem:
    exp(-α|r-A|²) * exp(-β|r-B|²) = K_AB * exp(-p|r-P|²)
    
    where p = α + β, P = (αA + βB)/p, K_AB = exp(-αβ|A-B|²/p)
    
    Args:
        prim1, prim2: Primitive GTOs
    
    Returns:
        Overlap integral value
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
    
    PA = P - A
    PB = P - B
    
    Sx = overlap_1d(la, lb, PA[0], PB[0], p)
    Sy = overlap_1d(ma, mb, PA[1], PB[1], p)
    Sz = overlap_1d(na, nb, PA[2], PB[2], p)
    
    prefactor = (np.pi / p) ** 1.5
    
    return prefactor * K_AB * Sx * Sy * Sz


def overlap_1d(l1: int, l2: int, PA: float, PB: float, p: float) -> float:
    """
    Compute 1D overlap integral using recursion relation.
    
    Obara-Saika recursion for overlap:
    S^{i+1,j} = (P-A) * S^{i,j} + (1/(2p)) * [i * S^{i-1,j} + j * S^{i,j-1}]
    S^{i,j+1} = (P-B) * S^{i,j} + (1/(2p)) * [i * S^{i-1,j} + j * S^{i,j-1}]
    
    Args:
        l1, l2: Angular momentum quantum numbers
        PA, PB: P-A and P-B distances
        p: Combined exponent
    
    Returns:
        1D overlap integral
    """
    S = np.zeros((l1 + 1, l2 + 1))
    
    S[0, 0] = 1.0
    
    if l2 > 0:
        S[0, 1] = PB
        for j in range(1, l2):
            S[0, j + 1] = PB * S[0, j] + j / (2.0 * p) * S[0, j - 1]
    
    for i in range(l1):
        S[i + 1, 0] = PA * S[i, 0]
        if i > 0:
            S[i + 1, 0] += i / (2.0 * p) * S[i - 1, 0]
        
        for j in range(l2 + 1):
            if j == 0:
                continue
            S[i + 1, j] = PA * S[i, j]
            if i > 0:
                S[i + 1, j] += i / (2.0 * p) * S[i - 1, j]
            if j > 0:
                S[i + 1, j] += j / (2.0 * p) * S[i, j - 1]
    
    return S[l1, l2]


def overlap_contracted(cgto1: 'ContractedGTO', cgto2: 'ContractedGTO') -> float:
    """
    Compute overlap integral between two contracted GTOs.
    
    S = sum_p sum_q d_p d_q S_pq
    
    Args:
        cgto1, cgto2: Contracted GTOs
    
    Returns:
        Overlap integral value
    """
    S = 0.0
    for prim1 in cgto1.primitives:
        for prim2 in cgto2.primitives:
            S += prim1.coeff * prim2.coeff * overlap_primitive(prim1, prim2)
    return S


def compute_overlap_matrix(basis: 'BasisSet') -> np.ndarray:
    """
    Compute overlap matrix S for entire basis set.
    
    S_μν = ∫ χ_μ(r) χ_ν(r) dr
    
    Args:
        basis: BasisSet object
    
    Returns:
        Overlap matrix (n_basis x n_basis)
    """
    n = basis.n_basis
    S = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1):
            S[i, j] = overlap_contracted(basis[i].cgto, basis[j].cgto)
            S[j, i] = S[i, j]
    
    return S
