"""
Bra-Ket Notation Implementation for Quantum Mechanics Education

This module provides a comprehensive implementation of Dirac's bra-ket notation
for quantum mechanics, designed for educational purposes and numerical computation.

Author: Devin AI Assistant
Purpose: Educational tool for understanding quantum mechanics notation
"""

import numpy as np
from typing import Union, Optional, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Ket:
    """
    Represents a quantum state vector |ψ⟩ (ket) in Dirac notation.
    
    A ket is a column vector in a complex Hilbert space.
    """
    
    def __init__(self, state: Union[np.ndarray, List[complex]]):
        """
        Initialize a ket state.
        
        Args:
            state: Complex array representing the quantum state
        """
        self.state = np.array(state, dtype=complex)
        if self.state.ndim != 1:
            raise ValueError("Ket state must be a 1D array")
    
    def __repr__(self) -> str:
        return f"Ket({self.state})"
    
    def __str__(self) -> str:
        return f"|ψ⟩ = {self.state}"
    
    def normalize(self) -> 'Ket':
        """Normalize the ket to unit length."""
        norm = np.sqrt(np.vdot(self.state, self.state).real)
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        return Ket(self.state / norm)
    
    def is_normalized(self, tolerance: float = 1e-10) -> bool:
        """Check if the ket is normalized."""
        norm_squared = np.vdot(self.state, self.state).real
        return abs(norm_squared - 1.0) < tolerance
    
    def bra(self) -> 'Bra':
        """Return the corresponding bra ⟨ψ|."""
        return Bra(self.state)
    
    def __add__(self, other: 'Ket') -> 'Ket':
        """Add two kets."""
        if not isinstance(other, Ket):
            raise TypeError("Can only add Ket to Ket")
        return Ket(self.state + other.state)
    
    def __sub__(self, other: 'Ket') -> 'Ket':
        """Subtract two kets."""
        if not isinstance(other, Ket):
            raise TypeError("Can only subtract Ket from Ket")
        return Ket(self.state - other.state)
    
    def __mul__(self, scalar: Union[complex, float, int]) -> 'Ket':
        """Multiply ket by scalar."""
        return Ket(scalar * self.state)
    
    def __rmul__(self, scalar: Union[complex, float, int]) -> 'Ket':
        """Right multiplication by scalar."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: Union[complex, float, int]) -> 'Ket':
        """Divide ket by scalar."""
        return Ket(self.state / scalar)
    
    def tensor_product(self, other: 'Ket') -> 'Ket':
        """Compute tensor product with another ket."""
        return Ket(np.kron(self.state, other.state))
    
    def __matmul__(self, other: 'Ket') -> 'Ket':
        """Tensor product using @ operator."""
        return self.tensor_product(other)


class Bra:
    """
    Represents a quantum state bra ⟨ψ| in Dirac notation.
    
    A bra is the complex conjugate transpose of a ket.
    """
    
    def __init__(self, state: Union[np.ndarray, List[complex]]):
        """
        Initialize a bra state.
        
        Args:
            state: Complex array representing the quantum state
        """
        self.state = np.array(state, dtype=complex)
        if self.state.ndim != 1:
            raise ValueError("Bra state must be a 1D array")
    
    def __repr__(self) -> str:
        return f"Bra({self.state.conj()})"
    
    def __str__(self) -> str:
        return f"⟨ψ| = {self.state.conj()}"
    
    def ket(self) -> Ket:
        """Return the corresponding ket |ψ⟩."""
        return Ket(self.state)
    
    def __mul__(self, other: Union[Ket, 'Operator']) -> Union[complex, 'Bra']:
        """
        Multiply bra with ket (inner product) or operator.
        
        Returns:
            complex: If multiplied with ket (inner product)
            Bra: If multiplied with operator
        """
        if isinstance(other, Ket):
            return np.vdot(self.state, other.state)
        elif isinstance(other, Operator):
            return Bra(self.state.conj() @ other.matrix.conj().T)
        else:
            raise TypeError("Bra can only multiply with Ket or Operator")


class Operator:
    """
    Represents a quantum operator (observable or transformation).
    """
    
    def __init__(self, matrix: Union[np.ndarray, List[List[complex]]]):
        """
        Initialize an operator.
        
        Args:
            matrix: 2D complex array representing the operator matrix
        """
        self.matrix = np.array(matrix, dtype=complex)
        if self.matrix.ndim != 2:
            raise ValueError("Operator must be a 2D matrix")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Operator matrix must be square")
    
    def __repr__(self) -> str:
        return f"Operator(\n{self.matrix}\n)"
    
    def __str__(self) -> str:
        return f"Operator:\n{self.matrix}"
    
    def is_hermitian(self, tolerance: float = 1e-10) -> bool:
        """Check if the operator is Hermitian (A† = A)."""
        return np.allclose(self.matrix, self.matrix.conj().T, atol=tolerance)
    
    def is_unitary(self, tolerance: float = 1e-10) -> bool:
        """Check if the operator is unitary (A†A = I)."""
        identity = np.eye(self.matrix.shape[0])
        product = self.matrix.conj().T @ self.matrix
        return np.allclose(product, identity, atol=tolerance)
    
    def dagger(self) -> 'Operator':
        """Return the Hermitian conjugate (adjoint) of the operator."""
        return Operator(self.matrix.conj().T)
    
    def eigenvalues_eigenvectors(self) -> Tuple[np.ndarray, List[Ket]]:
        """
        Compute eigenvalues and eigenvectors.
        
        Returns:
            Tuple of (eigenvalues, list of eigenvector kets)
        """
        if self.is_hermitian():
            eigenvals, eigenvecs = np.linalg.eigh(self.matrix)
        else:
            eigenvals, eigenvecs = np.linalg.eig(self.matrix)
        
        ket_eigenvecs = [Ket(eigenvecs[:, i]) for i in range(len(eigenvals))]
        
        return eigenvals, ket_eigenvecs
    
    def expectation_value(self, state: Ket) -> complex:
        """
        Compute expectation value ⟨ψ|A|ψ⟩.
        
        Args:
            state: Quantum state ket
            
        Returns:
            Expectation value of the operator in the given state
        """
        return np.vdot(state.state, self.matrix @ state.state)
    
    def __mul__(self, other: Union[Ket, 'Operator', complex, float, int]) -> Union[Ket, 'Operator']:
        """Multiply operator with ket, operator, or scalar."""
        if isinstance(other, Ket):
            return Ket(self.matrix @ other.state)
        elif isinstance(other, Operator):
            return Operator(self.matrix @ other.matrix)
        elif isinstance(other, (complex, float, int)):
            return Operator(other * self.matrix)
        else:
            raise TypeError("Operator can multiply with Ket, Operator, or scalar")
    
    def __rmul__(self, scalar: Union[complex, float, int]) -> 'Operator':
        """Right multiplication by scalar."""
        return Operator(scalar * self.matrix)
    
    def __truediv__(self, scalar: Union[complex, float, int]) -> 'Operator':
        """Divide operator by scalar."""
        return Operator(self.matrix / scalar)
    
    def __add__(self, other: 'Operator') -> 'Operator':
        """Add two operators."""
        if not isinstance(other, Operator):
            raise TypeError("Can only add Operator to Operator")
        return Operator(self.matrix + other.matrix)
    
    def __sub__(self, other: 'Operator') -> 'Operator':
        """Subtract two operators."""
        if not isinstance(other, Operator):
            raise TypeError("Can only subtract Operator from Operator")
        return Operator(self.matrix - other.matrix)
    
    def commutator(self, other: 'Operator') -> 'Operator':
        """Compute commutator [A, B] = AB - BA."""
        return Operator(self.matrix @ other.matrix - other.matrix @ self.matrix)
    
    def anticommutator(self, other: 'Operator') -> 'Operator':
        """Compute anticommutator {A, B} = AB + BA."""
        return Operator(self.matrix @ other.matrix + other.matrix @ self.matrix)


def outer_product(ket1: Ket, ket2: Ket) -> Operator:
    """
    Compute outer product |ψ⟩⟨φ| of two kets.
    
    Args:
        ket1: First ket |ψ⟩
        ket2: Second ket |φ⟩
        
    Returns:
        Operator representing the outer product
    """
    return Operator(np.outer(ket1.state, ket2.state.conj()))


def inner_product(bra: Bra, ket: Ket) -> complex:
    """
    Compute inner product ⟨φ|ψ⟩.
    
    Args:
        bra: Bra state ⟨φ|
        ket: Ket state |ψ⟩
        
    Returns:
        Complex inner product
    """
    return bra * ket


def projection_operator(ket: Ket) -> Operator:
    """
    Create projection operator |ψ⟩⟨ψ| for a given ket.
    
    Args:
        ket: Quantum state ket
        
    Returns:
        Projection operator
    """
    return outer_product(ket, ket)


class QuantumStates:
    """Collection of common quantum states."""
    
    @staticmethod
    def spin_up() -> Ket:
        """Spin-up state |↑⟩."""
        return Ket([1, 0])
    
    @staticmethod
    def spin_down() -> Ket:
        """Spin-down state |↓⟩."""
        return Ket([0, 1])
    
    @staticmethod
    def plus_state() -> Ket:
        """Plus state |+⟩ = (|↑⟩ + |↓⟩)/√2."""
        return Ket([1/np.sqrt(2), 1/np.sqrt(2)])
    
    @staticmethod
    def minus_state() -> Ket:
        """Minus state |-⟩ = (|↑⟩ - |↓⟩)/√2."""
        return Ket([1/np.sqrt(2), -1/np.sqrt(2)])
    
    @staticmethod
    def right_circular() -> Ket:
        """Right circular polarization |R⟩ = (|↑⟩ + i|↓⟩)/√2."""
        return Ket([1/np.sqrt(2), 1j/np.sqrt(2)])
    
    @staticmethod
    def left_circular() -> Ket:
        """Left circular polarization |L⟩ = (|↑⟩ - i|↓⟩)/√2."""
        return Ket([1/np.sqrt(2), -1j/np.sqrt(2)])


class PauliMatrices:
    """Collection of Pauli matrices and related operators."""
    
    @staticmethod
    def sigma_x() -> Operator:
        """Pauli-X matrix (bit-flip)."""
        return Operator([[0, 1], [1, 0]])
    
    @staticmethod
    def sigma_y() -> Operator:
        """Pauli-Y matrix."""
        return Operator([[0, -1j], [1j, 0]])
    
    @staticmethod
    def sigma_z() -> Operator:
        """Pauli-Z matrix (phase-flip)."""
        return Operator([[1, 0], [0, -1]])
    
    @staticmethod
    def identity() -> Operator:
        """2x2 Identity matrix."""
        return Operator([[1, 0], [0, 1]])
    
    @staticmethod
    def hadamard() -> Operator:
        """Hadamard gate."""
        return Operator([[1, 1], [1, -1]]) / np.sqrt(2)
    
    @staticmethod
    def phase_gate(phi: float) -> Operator:
        """Phase gate with phase phi."""
        return Operator([[1, 0], [0, np.exp(1j * phi)]])


class BlochSphere:
    """Utilities for Bloch sphere representation of spin-1/2 states."""
    
    @staticmethod
    def state_to_bloch_vector(ket: Ket) -> np.ndarray:
        """
        Convert a spin-1/2 state to Bloch vector coordinates.
        
        Args:
            ket: Normalized spin-1/2 state
            
        Returns:
            3D Bloch vector [x, y, z]
        """
        if len(ket.state) != 2:
            raise ValueError("Bloch sphere representation only valid for spin-1/2 (2D) states")
        
        if not ket.is_normalized():
            ket = ket.normalize()
        
        sigma_x = PauliMatrices.sigma_x()
        sigma_y = PauliMatrices.sigma_y()
        sigma_z = PauliMatrices.sigma_z()
        
        x = sigma_x.expectation_value(ket).real
        y = sigma_y.expectation_value(ket).real
        z = sigma_z.expectation_value(ket).real
        
        return np.array([x, y, z])
    
    @staticmethod
    def bloch_vector_to_state(bloch_vector: np.ndarray) -> Ket:
        """
        Convert Bloch vector to spin-1/2 state.
        
        Args:
            bloch_vector: 3D Bloch vector [x, y, z]
            
        Returns:
            Corresponding spin-1/2 state ket
        """
        x, y, z = bloch_vector
        
        norm = np.linalg.norm(bloch_vector)
        if norm > 1 + 1e-10:
            raise ValueError("Bloch vector must have norm ≤ 1")
        
        theta = np.arccos(z) if norm > 1e-10 else 0
        phi = np.arctan2(y, x) if abs(x) > 1e-10 or abs(y) > 1e-10 else 0
        
        cos_half_theta = np.cos(theta / 2)
        sin_half_theta = np.sin(theta / 2)
        
        state = np.array([
            cos_half_theta,
            sin_half_theta * np.exp(1j * phi)
        ])
        
        return Ket(state)
    
    @staticmethod
    def plot_bloch_sphere(states: List[Ket], labels: Optional[List[str]] = None, 
                         title: str = "Bloch Sphere") -> plt.Figure:
        """
        Plot states on the Bloch sphere.
        
        Args:
            states: List of spin-1/2 states to plot
            labels: Optional labels for each state
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')
        
        ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', alpha=0.3)
        ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', alpha=0.3)
        ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', alpha=0.3)
        
        ax.text(1.3, 0, 0, 'X', fontsize=12)
        ax.text(0, 1.3, 0, 'Y', fontsize=12)
        ax.text(0, 0, 1.3, 'Z', fontsize=12)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(states)))
        
        for i, state in enumerate(states):
            bloch_vec = BlochSphere.state_to_bloch_vector(state)
            x, y, z = bloch_vec
            
            ax.quiver(0, 0, 0, x, y, z, color=colors[i], arrow_length_ratio=0.1, linewidth=3)
            
            ax.scatter([x], [y], [z], color=colors[i], s=100)
            
            if labels and i < len(labels):
                ax.text(x*1.1, y*1.1, z*1.1, labels[i], fontsize=10)
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_title(title, fontsize=14)
        
        return fig


def demonstrate_basic_operations():
    """Demonstrate basic bra-ket operations."""
    print("=== Basic Bra-Ket Operations Demo ===\n")
    
    psi = QuantumStates.plus_state()
    phi = QuantumStates.spin_up()
    
    print(f"State |ψ⟩ = {psi}")
    print(f"State |φ⟩ = {phi}")
    print(f"Normalization check |ψ⟩: {psi.is_normalized()}")
    print()
    
    inner_prod = psi.bra() * phi
    print(f"Inner product ⟨ψ|φ⟩ = {inner_prod}")
    print(f"Probability |⟨ψ|φ⟩|² = {abs(inner_prod)**2}")
    print()
    
    proj = outer_product(psi, psi)
    print("Projection operator |ψ⟩⟨ψ|:")
    print(proj.matrix)
    print()
    
    sigma_x = PauliMatrices.sigma_x()
    sigma_y = PauliMatrices.sigma_y()
    sigma_z = PauliMatrices.sigma_z()
    
    print("Expectation values in |+⟩ state:")
    print(f"⟨σₓ⟩ = {sigma_x.expectation_value(psi)}")
    print(f"⟨σᵧ⟩ = {sigma_y.expectation_value(psi)}")
    print(f"⟨σᵤ⟩ = {sigma_z.expectation_value(psi)}")
    print()
    
    eigenvals, eigenvecs = sigma_z.eigenvalues_eigenvectors()
    print("σᵤ eigenvalues and eigenvectors:")
    for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs)):
        print(f"λ_{i} = {val}, |ψ_{i}⟩ = {vec.state}")
    print()


if __name__ == "__main__":
    demonstrate_basic_operations()
