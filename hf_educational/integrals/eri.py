"""
Electron repulsion integral (ERI) evaluation for Gaussian-type orbitals.

Implements analytical evaluation of two-electron integrals:
(μν|λσ) = ∫∫ χ_μ(r1) χ_ν(r1) |r1-r2|^(-1) χ_λ(r2) χ_σ(r2) dr1 dr2

Using Gaussian product theorem, Boys function, and Obara-Saika recursion.
"""

import numpy as np
from typing import TYPE_CHECKING
from .boys import boys_function_array

if TYPE_CHECKING:
    from basis.basis_set import PrimitiveGTO, ContractedGTO, BasisSet


def eri_primitive(prim1: 'PrimitiveGTO', prim2: 'PrimitiveGTO',
                  prim3: 'PrimitiveGTO', prim4: 'PrimitiveGTO') -> float:
    """
    Compute electron repulsion integral between four primitive GTOs.
    
    (ab|cd) = ∫∫ g_a(r1) g_b(r1) |r1-r2|^(-1) g_c(r2) g_d(r2) dr1 dr2
    
    Uses Gaussian product theorem twice and Boys function.
    
    Args:
        prim1, prim2, prim3, prim4: Primitive GTOs
    
    Returns:
        ERI value
    """
    alpha_a = prim1.alpha
    alpha_b = prim2.alpha
    alpha_c = prim3.alpha
    alpha_d = prim4.alpha
    
    A = prim1.center
    B = prim2.center
    C = prim3.center
    D = prim4.center
    
    la, ma, na = prim1.l, prim1.m, prim1.n
    lb, mb, nb = prim2.l, prim2.m, prim2.n
    lc, mc, nc = prim3.l, prim3.m, prim3.n
    ld, md, nd = prim4.l, prim4.m, prim4.n
    
    p = alpha_a + alpha_b
    q = alpha_c + alpha_d
    
    mu_ab = alpha_a * alpha_b / p
    mu_cd = alpha_c * alpha_d / q
    
    AB = A - B
    CD = C - D
    AB2 = np.dot(AB, AB)
    CD2 = np.dot(CD, CD)
    
    P = (alpha_a * A + alpha_b * B) / p
    Q = (alpha_c * C + alpha_d * D) / q
    
    K_AB = np.exp(-mu_ab * AB2)
    K_CD = np.exp(-mu_cd * CD2)
    
    PQ = P - Q
    PQ2 = np.dot(PQ, PQ)
    
    alpha = p * q / (p + q)
    t = alpha * PQ2
    
    L_max = la + lb + lc + ld + ma + mb + mc + md + na + nb + nc + nd
    
    F = boys_function_array(L_max, t)
    
    eri_val = eri_recursive(la, lb, lc, ld,
                           ma, mb, mc, md,
                           na, nb, nc, nd,
                           P - A, P - B, Q - C, Q - D, P - Q,
                           p, q, alpha, F, 0)
    
    prefactor = 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q)) * K_AB * K_CD
    
    return prefactor * eri_val


def eri_recursive(la: int, lb: int, lc: int, ld: int,
                 ma: int, mb: int, mc: int, md: int,
                 na: int, nb: int, nc: int, nd: int,
                 PA: np.ndarray, PB: np.ndarray, QC: np.ndarray, QD: np.ndarray, PQ: np.ndarray,
                 p: float, q: float, alpha: float, F: np.ndarray, n: int) -> float:
    """
    Recursive evaluation of ERI auxiliary integrals.
    
    Uses Obara-Saika recursion relations for electron repulsion.
    This is a simplified implementation for educational purposes.
    
    Args:
        la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd: Angular momentum quantum numbers
        PA, PB, QC, QD, PQ: Distance vectors
        p, q, alpha: Combined exponents
        F: Array of Boys function values
        n: Boys function order index
    
    Returns:
        Auxiliary integral value
    """
    L = la + lb + lc + ld + ma + mb + mc + md + na + nb + nc + nd
    
    if L == 0:
        return F[n]
    
    zeta = p + q
    
    if la > 0:
        term1 = PA[0] * eri_recursive(la - 1, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                                      PA, PB, QC, QD, PQ, p, q, alpha, F, n)
        
        term2 = -(q / zeta) * PQ[0] * eri_recursive(la - 1, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                                                     PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        result = term1 + term2
        
        if la > 1:
            coeff = (la - 1) / (2.0 * p)
            result += coeff * (eri_recursive(la - 2, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                                            PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (q / zeta) * eri_recursive(la - 2, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        if lb > 0:
            coeff = lb / (2.0 * p)
            result += coeff * (eri_recursive(la - 1, lb - 1, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                                            PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (q / zeta) * eri_recursive(la - 1, lb - 1, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        if lc > 0:
            coeff = lc / (2.0 * zeta)
            result += coeff * eri_recursive(la - 1, lb, lc - 1, ld, ma, mb, mc, md, na, nb, nc, nd,
                                           PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        if ld > 0:
            coeff = ld / (2.0 * zeta)
            result += coeff * eri_recursive(la - 1, lb, lc, ld - 1, ma, mb, mc, md, na, nb, nc, nd,
                                           PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        return result
    
    if lb > 0:
        term1 = PB[0] * eri_recursive(la, lb - 1, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                                      PA, PB, QC, QD, PQ, p, q, alpha, F, n)
        
        term2 = -(q / zeta) * PQ[0] * eri_recursive(la, lb - 1, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                                                     PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        result = term1 + term2
        
        if lb > 1:
            coeff = (lb - 1) / (2.0 * p)
            result += coeff * (eri_recursive(la, lb - 2, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                                            PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (q / zeta) * eri_recursive(la, lb - 2, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        if lc > 0:
            coeff = lc / (2.0 * zeta)
            result += coeff * eri_recursive(la, lb - 1, lc - 1, ld, ma, mb, mc, md, na, nb, nc, nd,
                                           PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        if ld > 0:
            coeff = ld / (2.0 * zeta)
            result += coeff * eri_recursive(la, lb - 1, lc, ld - 1, ma, mb, mc, md, na, nb, nc, nd,
                                           PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        return result
    
    if lc > 0:
        term1 = QC[0] * eri_recursive(la, lb, lc - 1, ld, ma, mb, mc, md, na, nb, nc, nd,
                                      PA, PB, QC, QD, PQ, p, q, alpha, F, n)
        
        term2 = (p / zeta) * PQ[0] * eri_recursive(la, lb, lc - 1, ld, ma, mb, mc, md, na, nb, nc, nd,
                                                    PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        result = term1 + term2
        
        if lc > 1:
            coeff = (lc - 1) / (2.0 * q)
            result += coeff * (eri_recursive(la, lb, lc - 2, ld, ma, mb, mc, md, na, nb, nc, nd,
                                            PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (p / zeta) * eri_recursive(la, lb, lc - 2, ld, ma, mb, mc, md, na, nb, nc, nd,
                                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        if ld > 0:
            coeff = ld / (2.0 * q)
            result += coeff * (eri_recursive(la, lb, lc - 1, ld - 1, ma, mb, mc, md, na, nb, nc, nd,
                                            PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (p / zeta) * eri_recursive(la, lb, lc - 1, ld - 1, ma, mb, mc, md, na, nb, nc, nd,
                                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        return result
    
    if ld > 0:
        term1 = QD[0] * eri_recursive(la, lb, lc, ld - 1, ma, mb, mc, md, na, nb, nc, nd,
                                      PA, PB, QC, QD, PQ, p, q, alpha, F, n)
        
        term2 = (p / zeta) * PQ[0] * eri_recursive(la, lb, lc, ld - 1, ma, mb, mc, md, na, nb, nc, nd,
                                                    PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        result = term1 + term2
        
        if ld > 1:
            coeff = (ld - 1) / (2.0 * q)
            result += coeff * (eri_recursive(la, lb, lc, ld - 2, ma, mb, mc, md, na, nb, nc, nd,
                                            PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (p / zeta) * eri_recursive(la, lb, lc, ld - 2, ma, mb, mc, md, na, nb, nc, nd,
                                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        return result
    
    if ma > 0 or mb > 0 or mc > 0 or md > 0:
        return eri_recursive_y(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                              PA, PB, QC, QD, PQ, p, q, alpha, F, n)
    
    if na > 0 or nb > 0 or nc > 0 or nd > 0:
        return eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                              PA, PB, QC, QD, PQ, p, q, alpha, F, n)
    
    return F[n]


def eri_recursive_y(la: int, lb: int, lc: int, ld: int,
                   ma: int, mb: int, mc: int, md: int,
                   na: int, nb: int, nc: int, nd: int,
                   PA: np.ndarray, PB: np.ndarray, QC: np.ndarray, QD: np.ndarray, PQ: np.ndarray,
                   p: float, q: float, alpha: float, F: np.ndarray, n: int) -> float:
    """Recursion for y-component (similar to x-component)."""
    zeta = p + q
    
    if ma > 0:
        term1 = PA[1] * eri_recursive_y(la, lb, lc, ld, ma - 1, mb, mc, md, na, nb, nc, nd,
                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n)
        
        term2 = -(q / zeta) * PQ[1] * eri_recursive_y(la, lb, lc, ld, ma - 1, mb, mc, md, na, nb, nc, nd,
                                                       PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        result = term1 + term2
        
        if ma > 1:
            coeff = (ma - 1) / (2.0 * p)
            result += coeff * (eri_recursive_y(la, lb, lc, ld, ma - 2, mb, mc, md, na, nb, nc, nd,
                                              PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (q / zeta) * eri_recursive_y(la, lb, lc, ld, ma - 2, mb, mc, md, na, nb, nc, nd,
                                                          PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        if mb > 0:
            coeff = mb / (2.0 * p)
            result += coeff * (eri_recursive_y(la, lb, lc, ld, ma - 1, mb - 1, mc, md, na, nb, nc, nd,
                                              PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (q / zeta) * eri_recursive_y(la, lb, lc, ld, ma - 1, mb - 1, mc, md, na, nb, nc, nd,
                                                          PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        if mc > 0:
            coeff = mc / (2.0 * zeta)
            result += coeff * eri_recursive_y(la, lb, lc, ld, ma - 1, mb, mc - 1, md, na, nb, nc, nd,
                                             PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        if md > 0:
            coeff = md / (2.0 * zeta)
            result += coeff * eri_recursive_y(la, lb, lc, ld, ma - 1, mb, mc, md - 1, na, nb, nc, nd,
                                             PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        return result
    
    if mb > 0:
        term1 = PB[1] * eri_recursive_y(la, lb, lc, ld, ma, mb - 1, mc, md, na, nb, nc, nd,
                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n)
        
        term2 = -(q / zeta) * PQ[1] * eri_recursive_y(la, lb, lc, ld, ma, mb - 1, mc, md, na, nb, nc, nd,
                                                       PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        result = term1 + term2
        
        if mb > 1:
            coeff = (mb - 1) / (2.0 * p)
            result += coeff * (eri_recursive_y(la, lb, lc, ld, ma, mb - 2, mc, md, na, nb, nc, nd,
                                              PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (q / zeta) * eri_recursive_y(la, lb, lc, ld, ma, mb - 2, mc, md, na, nb, nc, nd,
                                                          PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        if mc > 0:
            coeff = mc / (2.0 * zeta)
            result += coeff * eri_recursive_y(la, lb, lc, ld, ma, mb - 1, mc - 1, md, na, nb, nc, nd,
                                             PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        if md > 0:
            coeff = md / (2.0 * zeta)
            result += coeff * eri_recursive_y(la, lb, lc, ld, ma, mb - 1, mc, md - 1, na, nb, nc, nd,
                                             PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        return result
    
    if mc > 0:
        term1 = QC[1] * eri_recursive_y(la, lb, lc, ld, ma, mb, mc - 1, md, na, nb, nc, nd,
                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n)
        
        term2 = (p / zeta) * PQ[1] * eri_recursive_y(la, lb, lc, ld, ma, mb, mc - 1, md, na, nb, nc, nd,
                                                      PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        result = term1 + term2
        
        if mc > 1:
            coeff = (mc - 1) / (2.0 * q)
            result += coeff * (eri_recursive_y(la, lb, lc, ld, ma, mb, mc - 2, md, na, nb, nc, nd,
                                              PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (p / zeta) * eri_recursive_y(la, lb, lc, ld, ma, mb, mc - 2, md, na, nb, nc, nd,
                                                          PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        if md > 0:
            coeff = md / (2.0 * q)
            result += coeff * (eri_recursive_y(la, lb, lc, ld, ma, mb, mc - 1, md - 1, na, nb, nc, nd,
                                              PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (p / zeta) * eri_recursive_y(la, lb, lc, ld, ma, mb, mc - 1, md - 1, na, nb, nc, nd,
                                                          PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        return result
    
    if md > 0:
        term1 = QD[1] * eri_recursive_y(la, lb, lc, ld, ma, mb, mc, md - 1, na, nb, nc, nd,
                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n)
        
        term2 = (p / zeta) * PQ[1] * eri_recursive_y(la, lb, lc, ld, ma, mb, mc, md - 1, na, nb, nc, nd,
                                                      PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        result = term1 + term2
        
        if md > 1:
            coeff = (md - 1) / (2.0 * q)
            result += coeff * (eri_recursive_y(la, lb, lc, ld, ma, mb, mc, md - 2, na, nb, nc, nd,
                                              PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (p / zeta) * eri_recursive_y(la, lb, lc, ld, ma, mb, mc, md - 2, na, nb, nc, nd,
                                                          PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        return result
    
    if na > 0 or nb > 0 or nc > 0 or nd > 0:
        return eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                              PA, PB, QC, QD, PQ, p, q, alpha, F, n)
    
    return F[n]


def eri_recursive_z(la: int, lb: int, lc: int, ld: int,
                   ma: int, mb: int, mc: int, md: int,
                   na: int, nb: int, nc: int, nd: int,
                   PA: np.ndarray, PB: np.ndarray, QC: np.ndarray, QD: np.ndarray, PQ: np.ndarray,
                   p: float, q: float, alpha: float, F: np.ndarray, n: int) -> float:
    """Recursion for z-component (similar to x-component)."""
    zeta = p + q
    
    if na > 0:
        term1 = PA[2] * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na - 1, nb, nc, nd,
                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n)
        
        term2 = -(q / zeta) * PQ[2] * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na - 1, nb, nc, nd,
                                                       PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        result = term1 + term2
        
        if na > 1:
            coeff = (na - 1) / (2.0 * p)
            result += coeff * (eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na - 2, nb, nc, nd,
                                              PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (q / zeta) * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na - 2, nb, nc, nd,
                                                          PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        if nb > 0:
            coeff = nb / (2.0 * p)
            result += coeff * (eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na - 1, nb - 1, nc, nd,
                                              PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (q / zeta) * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na - 1, nb - 1, nc, nd,
                                                          PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        if nc > 0:
            coeff = nc / (2.0 * zeta)
            result += coeff * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na - 1, nb, nc - 1, nd,
                                             PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        if nd > 0:
            coeff = nd / (2.0 * zeta)
            result += coeff * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na - 1, nb, nc, nd - 1,
                                             PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        return result
    
    if nb > 0:
        term1 = PB[2] * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb - 1, nc, nd,
                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n)
        
        term2 = -(q / zeta) * PQ[2] * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb - 1, nc, nd,
                                                       PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        result = term1 + term2
        
        if nb > 1:
            coeff = (nb - 1) / (2.0 * p)
            result += coeff * (eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb - 2, nc, nd,
                                              PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (q / zeta) * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb - 2, nc, nd,
                                                          PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        if nc > 0:
            coeff = nc / (2.0 * zeta)
            result += coeff * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb - 1, nc - 1, nd,
                                             PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        if nd > 0:
            coeff = nd / (2.0 * zeta)
            result += coeff * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb - 1, nc, nd - 1,
                                             PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        return result
    
    if nc > 0:
        term1 = QC[2] * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc - 1, nd,
                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n)
        
        term2 = (p / zeta) * PQ[2] * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc - 1, nd,
                                                      PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        result = term1 + term2
        
        if nc > 1:
            coeff = (nc - 1) / (2.0 * q)
            result += coeff * (eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc - 2, nd,
                                              PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (p / zeta) * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc - 2, nd,
                                                          PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        if nd > 0:
            coeff = nd / (2.0 * q)
            result += coeff * (eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc - 1, nd - 1,
                                              PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (p / zeta) * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc - 1, nd - 1,
                                                          PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        return result
    
    if nd > 0:
        term1 = QD[2] * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd - 1,
                                        PA, PB, QC, QD, PQ, p, q, alpha, F, n)
        
        term2 = (p / zeta) * PQ[2] * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd - 1,
                                                      PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1)
        
        result = term1 + term2
        
        if nd > 1:
            coeff = (nd - 1) / (2.0 * q)
            result += coeff * (eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd - 2,
                                              PA, PB, QC, QD, PQ, p, q, alpha, F, n) -
                              (p / zeta) * eri_recursive_z(la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd - 2,
                                                          PA, PB, QC, QD, PQ, p, q, alpha, F, n + 1))
        
        return result
    
    return F[n]


def eri_contracted(cgto1: 'ContractedGTO', cgto2: 'ContractedGTO',
                  cgto3: 'ContractedGTO', cgto4: 'ContractedGTO') -> float:
    """
    Compute ERI between four contracted GTOs.
    
    (ab|cd) = sum_p sum_q sum_r sum_s d_p d_q d_r d_s (pq|rs)
    
    Args:
        cgto1, cgto2, cgto3, cgto4: Contracted GTOs
    
    Returns:
        ERI value
    """
    eri_val = 0.0
    for prim1 in cgto1.primitives:
        for prim2 in cgto2.primitives:
            for prim3 in cgto3.primitives:
                for prim4 in cgto4.primitives:
                    eri_val += (prim1.coeff * prim2.coeff * prim3.coeff * prim4.coeff *
                               eri_primitive(prim1, prim2, prim3, prim4))
    return eri_val


def compute_eri_tensor(basis: 'BasisSet', schwarz_threshold: float = 1e-10) -> np.ndarray:
    """
    Compute full ERI tensor for entire basis set.
    
    (μν|λσ) for all μ, ν, λ, σ
    
    Uses 8-fold symmetry and Schwarz screening.
    
    Args:
        basis: BasisSet object
        schwarz_threshold: Threshold for Schwarz screening
    
    Returns:
        ERI tensor (n_basis x n_basis x n_basis x n_basis)
    """
    n = basis.n_basis
    eri = np.zeros((n, n, n, n))
    
    print(f"Computing ERI tensor for {n} basis functions...")
    print(f"Total unique integrals (with 8-fold symmetry): ~{n**4 // 8}")
    
    schwarz = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            val = eri_contracted(basis[i].cgto, basis[j].cgto,
                                basis[i].cgto, basis[j].cgto)
            schwarz[i, j] = np.sqrt(abs(val))
            schwarz[j, i] = schwarz[i, j]
    
    n_computed = 0
    n_screened = 0
    
    for i in range(n):
        for j in range(i + 1):
            ij_schwarz = schwarz[i, j]
            
            for k in range(n):
                for l in range(k + 1):
                    kl_schwarz = schwarz[k, l]
                    
                    if ij_schwarz * kl_schwarz < schwarz_threshold:
                        n_screened += 1
                        continue
                    
                    val = eri_contracted(basis[i].cgto, basis[j].cgto,
                                        basis[k].cgto, basis[l].cgto)
                    
                    eri[i, j, k, l] = val
                    eri[j, i, k, l] = val
                    eri[i, j, l, k] = val
                    eri[j, i, l, k] = val
                    eri[k, l, i, j] = val
                    eri[l, k, i, j] = val
                    eri[k, l, j, i] = val
                    eri[l, k, j, i] = val
                    
                    n_computed += 1
    
    print(f"Computed: {n_computed}, Screened: {n_screened}")
    print(f"Screening efficiency: {100.0 * n_screened / (n_computed + n_screened):.1f}%")
    
    return eri
