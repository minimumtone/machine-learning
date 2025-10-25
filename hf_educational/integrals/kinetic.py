"""
Kinetic energy integral evaluation for Gaussian-type orbitals.

Implements analytical evaluation of kinetic energy integrals:
T_μν = ∫ χ_μ(r) (-1/2 ∇²) χ_ν(r) dr

Using Gaussian product theorem and analytical formulas.
"""

import numpy as np
from typing import TYPE_CHECKING
from .overlap import overlap_primitive, overlap_1d

if TYPE_CHECKING:
    from basis.basis_set import PrimitiveGTO, ContractedGTO, BasisSet


def kinetic_primitive(prim1: 'PrimitiveGTO', prim2: 'PrimitiveGTO') -> float:
    """
    Compute kinetic energy integral between two primitive GTOs.
    
    T = -1/2 ∫ χ_a ∇² χ_b dr
    
    Using the relation:
    ∇² [x^l y^m z^n exp(-α r²)] involves second derivatives
    
    T can be expressed in terms of overlap integrals with modified angular momenta.
    
    Args:
        prim1, prim2: Primitive GTOs
    
    Returns:
        Kinetic energy integral value
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
    
    prefactor = (np.pi / p) ** 1.5 * K_AB
    
    Sx = overlap_1d(la, lb, PA[0], PB[0], p)
    Sy = overlap_1d(ma, mb, PA[1], PB[1], p)
    Sz = overlap_1d(na, nb, PA[2], PB[2], p)
    
    term1 = beta * (2 * (lb + mb + nb) + 3) * Sx * Sy * Sz
    
    term2 = 0.0
    Sx_p2 = overlap_1d(la, lb + 2, PA[0], PB[0], p)
    term2 += 2.0 * beta**2 * Sx_p2 * Sy * Sz
    
    Sy_p2 = overlap_1d(ma, mb + 2, PA[1], PB[1], p)
    term2 += 2.0 * beta**2 * Sx * Sy_p2 * Sz
    
    Sz_p2 = overlap_1d(na, nb + 2, PA[2], PB[2], p)
    term2 += 2.0 * beta**2 * Sx * Sy * Sz_p2
    
    term3 = 0.0
    if lb >= 2:
        Sx_m2 = overlap_1d(la, lb - 2, PA[0], PB[0], p)
        term3 += 0.5 * lb * (lb - 1) * Sx_m2 * Sy * Sz
    
    if mb >= 2:
        Sy_m2 = overlap_1d(ma, mb - 2, PA[1], PB[1], p)
        term3 += 0.5 * mb * (mb - 1) * Sx * Sy_m2 * Sz
    
    if nb >= 2:
        Sz_m2 = overlap_1d(na, nb - 2, PA[2], PB[2], p)
        term3 += 0.5 * nb * (nb - 1) * Sx * Sy * Sz_m2
    
    T = term1 - term2 - term3
    
    return prefactor * T


def kinetic_contracted(cgto1: 'ContractedGTO', cgto2: 'ContractedGTO') -> float:
    """
    Compute kinetic energy integral between two contracted GTOs.
    
    T = sum_p sum_q d_p d_q T_pq
    
    Args:
        cgto1, cgto2: Contracted GTOs
    
    Returns:
        Kinetic energy integral value
    """
    T = 0.0
    for prim1 in cgto1.primitives:
        for prim2 in cgto2.primitives:
            T += prim1.coeff * prim2.coeff * kinetic_primitive(prim1, prim2)
    return T


def compute_kinetic_matrix(basis: 'BasisSet') -> np.ndarray:
    """
    Compute kinetic energy matrix T for entire basis set.
    
    T_μν = ∫ χ_μ(r) (-1/2 ∇²) χ_ν(r) dr
    
    Args:
        basis: BasisSet object
    
    Returns:
        Kinetic energy matrix (n_basis x n_basis)
    """
    n = basis.n_basis
    T = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1):
            T[i, j] = kinetic_contracted(basis[i].cgto, basis[j].cgto)
            T[j, i] = T[i, j]
    
    return T
