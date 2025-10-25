"""
Direct Inversion in the Iterative Subspace (DIIS) for SCF convergence acceleration.

DIIS extrapolates the Fock matrix using a linear combination of previous
Fock matrices and error vectors to minimize the error.

Reference: Pulay, P. Chem. Phys. Lett. 1980, 73, 393-398.
"""

import numpy as np
from typing import List


class DIIS:
    """
    DIIS convergence accelerator.
    
    Maintains a history of Fock matrices and error vectors,
    then finds optimal linear combination to minimize error.
    """
    
    def __init__(self, max_size: int = 8):
        """
        Initialize DIIS.
        
        Args:
            max_size: Maximum number of vectors to store
        """
        self.max_size = max_size
        self.fock_history: List[np.ndarray] = []
        self.error_history: List[np.ndarray] = []
    
    def update(self, F: np.ndarray, error: np.ndarray) -> np.ndarray:
        """
        Update DIIS and return extrapolated Fock matrix.
        
        Error vector is typically: e = FPS - SPF (commutator)
        
        Args:
            F: Current Fock matrix
            error: Current error vector
        
        Returns:
            Extrapolated Fock matrix
        """
        self.fock_history.append(F.copy())
        self.error_history.append(error.copy())
        
        if len(self.fock_history) > self.max_size:
            self.fock_history.pop(0)
            self.error_history.pop(0)
        
        if len(self.fock_history) < 2:
            return F
        
        n = len(self.fock_history)
        
        B = np.zeros((n + 1, n + 1))
        
        for i in range(n):
            for j in range(n):
                B[i, j] = np.sum(self.error_history[i] * self.error_history[j])
        
        B[n, :n] = -1.0
        B[:n, n] = -1.0
        B[n, n] = 0.0
        
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        
        try:
            coeffs = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return F
        
        F_diis = np.zeros_like(F)
        for i in range(n):
            F_diis += coeffs[i] * self.fock_history[i]
        
        return F_diis
    
    def reset(self):
        """Clear DIIS history."""
        self.fock_history.clear()
        self.error_history.clear()


def test_diis():
    """Test DIIS implementation."""
    print("Testing DIIS...")
    
    n = 3
    diis = DIIS(max_size=4)
    
    F_exact = np.array([
        [1.0, 0.1, 0.05],
        [0.1, 2.0, 0.15],
        [0.05, 0.15, 3.0]
    ])
    
    F = F_exact + 0.5 * np.random.randn(n, n)
    F = 0.5 * (F + F.T)
    
    print("\nTarget Fock matrix:")
    print(F_exact)
    
    print("\nInitial Fock matrix:")
    print(F)
    
    for iteration in range(10):
        error = F - F_exact
        error_norm = np.linalg.norm(error)
        
        print(f"\nIteration {iteration + 1}: Error norm = {error_norm:.6f}")
        
        F = diis.update(F, error)
        
        F = 0.7 * F + 0.3 * F_exact
    
    print("\nFinal Fock matrix:")
    print(F)
    print(f"\nFinal error: {np.linalg.norm(F - F_exact):.2e}")


if __name__ == "__main__":
    test_diis()
