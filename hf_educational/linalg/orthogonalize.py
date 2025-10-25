"""
Orthogonalization methods for basis set transformations.

Implements symmetric (Löwdin) and canonical orthogonalization
to transform non-orthogonal AO basis to orthogonal basis.
"""

import numpy as np
from scipy import linalg


def symmetric_orthogonalization(S: np.ndarray, threshold: float = 1e-7) -> np.ndarray:
    """
    Symmetric (Löwdin) orthogonalization: X = S^(-1/2)
    
    Computes S^(-1/2) via eigenvalue decomposition:
    S = U s U^T
    S^(-1/2) = U s^(-1/2) U^T
    
    Small eigenvalues (< threshold) are removed for numerical stability.
    
    Args:
        S: Overlap matrix (n x n)
        threshold: Eigenvalue threshold for numerical stability
    
    Returns:
        Transformation matrix X (n x n)
    """
    s, U = linalg.eigh(S)
    
    n_removed = np.sum(s < threshold)
    if n_removed > 0:
        print(f"Warning: Removed {n_removed} small eigenvalues (< {threshold})")
    
    s_inv_sqrt = np.zeros_like(s)
    mask = s >= threshold
    s_inv_sqrt[mask] = 1.0 / np.sqrt(s[mask])
    
    X = U @ np.diag(s_inv_sqrt) @ U.T
    
    identity_check = X.T @ S @ X
    error = np.max(np.abs(identity_check - np.eye(len(S))))
    if error > 1e-6:
        print(f"Warning: Orthogonalization error = {error:.2e}")
    
    return X


def canonical_orthogonalization(S: np.ndarray, threshold: float = 1e-7) -> np.ndarray:
    """
    Canonical orthogonalization using Cholesky decomposition.
    
    S = L L^T
    X = L^(-1)
    
    This is more numerically stable but breaks symmetry.
    
    Args:
        S: Overlap matrix (n x n)
        threshold: Not used, kept for API compatibility
    
    Returns:
        Transformation matrix X (n x n)
    """
    try:
        L = linalg.cholesky(S, lower=True)
        X = linalg.solve_triangular(L, np.eye(len(S)), lower=True).T
        
        identity_check = X.T @ S @ X
        error = np.max(np.abs(identity_check - np.eye(len(S))))
        if error > 1e-6:
            print(f"Warning: Orthogonalization error = {error:.2e}")
        
        return X
    except linalg.LinAlgError:
        print("Warning: Cholesky decomposition failed, falling back to symmetric orthogonalization")
        return symmetric_orthogonalization(S, threshold)


def test_orthogonalization():
    """Test orthogonalization methods."""
    print("Testing orthogonalization methods...")
    
    S = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ])
    
    print("\nOverlap matrix S:")
    print(S)
    
    X_sym = symmetric_orthogonalization(S)
    print("\nSymmetric orthogonalization X:")
    print(X_sym)
    
    identity = X_sym.T @ S @ X_sym
    print("\nX^T S X (should be identity):")
    print(identity)
    print(f"Max error: {np.max(np.abs(identity - np.eye(3))):.2e}")
    
    X_can = canonical_orthogonalization(S)
    print("\nCanonical orthogonalization X:")
    print(X_can)
    
    identity = X_can.T @ S @ X_can
    print("\nX^T S X (should be identity):")
    print(identity)
    print(f"Max error: {np.max(np.abs(identity - np.eye(3))):.2e}")


if __name__ == "__main__":
    test_orthogonalization()
