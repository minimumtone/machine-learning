"""
Eigenvalue problem solvers for HF equations.

Solves generalized eigenvalue problem F C = S C ε
"""

import numpy as np
from scipy import linalg


def solve_generalized_eigenproblem(F: np.ndarray, S: np.ndarray, 
                                   X: np.ndarray = None) -> tuple:
    """
    Solve generalized eigenvalue problem F C = S C ε
    
    Two methods:
    1. If X provided: Transform to orthogonal basis
       F' = X^T F X, solve F' C' = C' ε, then C = X C'
    2. Otherwise: Use scipy's eigh with generalized mode
    
    Args:
        F: Fock matrix (n x n)
        S: Overlap matrix (n x n)
        X: Orthogonalization matrix (optional)
    
    Returns:
        (eigenvalues, eigenvectors) tuple
        eigenvalues: MO energies (n,)
        eigenvectors: MO coefficients (n x n)
    """
    if X is not None:
        F_prime = X.T @ F @ X
        
        eps, C_prime = linalg.eigh(F_prime)
        
        C = X @ C_prime
    else:
        eps, C = linalg.eigh(F, S)
    
    idx = np.argsort(eps)
    eps = eps[idx]
    C = C[:, idx]
    
    return eps, C


def test_generalized_eigenproblem():
    """Test generalized eigenvalue problem solver."""
    print("Testing generalized eigenvalue problem solver...")
    
    F = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 3.0]
    ])
    
    S = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ])
    
    print("\nFock matrix F:")
    print(F)
    print("\nOverlap matrix S:")
    print(S)
    
    from .orthogonalize import symmetric_orthogonalization
    X = symmetric_orthogonalization(S)
    
    eps1, C1 = solve_generalized_eigenproblem(F, S, X)
    print("\nMethod 1 (with orthogonalization):")
    print(f"Eigenvalues: {eps1}")
    
    eps2, C2 = solve_generalized_eigenproblem(F, S, None)
    print("\nMethod 2 (direct):")
    print(f"Eigenvalues: {eps2}")
    
    print(f"\nDifference in eigenvalues: {np.max(np.abs(eps1 - eps2)):.2e}")
    
    residual = F @ C1 - S @ C1 @ np.diag(eps1)
    print(f"Residual norm (method 1): {np.linalg.norm(residual):.2e}")
    
    residual = F @ C2 - S @ C2 @ np.diag(eps2)
    print(f"Residual norm (method 2): {np.linalg.norm(residual):.2e}")


if __name__ == "__main__":
    test_generalized_eigenproblem()
