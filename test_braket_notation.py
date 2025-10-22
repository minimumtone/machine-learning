"""
Unit tests for bra-ket notation implementation.

This test suite verifies all operations of the braket_notation module.
"""

import unittest
import numpy as np
from braket_notation import (
    Ket, Bra, Operator, 
    outer_product, inner_product, projection_operator,
    QuantumStates, PauliMatrices, BlochSphere
)


class TestKet(unittest.TestCase):
    """Test cases for Ket class."""
    
    def test_ket_creation(self):
        """Test ket creation and basic properties."""
        psi = Ket([1, 0])
        self.assertEqual(len(psi.state), 2)
        self.assertEqual(psi.state[0], 1)
        self.assertEqual(psi.state[1], 0)
    
    def test_ket_normalization(self):
        """Test ket normalization."""
        psi = Ket([3, 4])
        psi_normalized = psi.normalize()
        self.assertTrue(psi_normalized.is_normalized())
        self.assertAlmostEqual(abs(psi_normalized.state[0]), 0.6)
        self.assertAlmostEqual(abs(psi_normalized.state[1]), 0.8)
    
    def test_ket_addition(self):
        """Test ket addition."""
        psi1 = Ket([1, 0])
        psi2 = Ket([0, 1])
        psi_sum = psi1 + psi2
        np.testing.assert_array_equal(psi_sum.state, [1, 1])
    
    def test_ket_scalar_multiplication(self):
        """Test scalar multiplication."""
        psi = Ket([1, 2])
        psi_scaled = 3 * psi
        np.testing.assert_array_equal(psi_scaled.state, [3, 6])
        
        psi_scaled2 = psi * 3
        np.testing.assert_array_equal(psi_scaled2.state, [3, 6])
    
    def test_ket_to_bra(self):
        """Test conversion from ket to bra."""
        psi = Ket([1, 1j])
        bra = psi.bra()
        self.assertIsInstance(bra, Bra)
        np.testing.assert_array_equal(bra.state, psi.state)
    
    def test_tensor_product(self):
        """Test tensor product of kets."""
        psi1 = Ket([1, 0])
        psi2 = Ket([0, 1])
        psi_tensor = psi1.tensor_product(psi2)
        np.testing.assert_array_equal(psi_tensor.state, [0, 1, 0, 0])


class TestBra(unittest.TestCase):
    """Test cases for Bra class."""
    
    def test_bra_creation(self):
        """Test bra creation."""
        bra = Bra([1, 0])
        self.assertEqual(len(bra.state), 2)
    
    def test_bra_to_ket(self):
        """Test conversion from bra to ket."""
        bra = Bra([1, 1j])
        ket = bra.ket()
        self.assertIsInstance(ket, Ket)
        np.testing.assert_array_equal(ket.state, bra.state)


class TestInnerProduct(unittest.TestCase):
    """Test cases for inner product operations."""
    
    def test_inner_product_orthogonal(self):
        """Test inner product of orthogonal states."""
        psi = Ket([1, 0])
        phi = Ket([0, 1])
        result = psi.bra() * phi
        self.assertAlmostEqual(result, 0)
    
    def test_inner_product_same_state(self):
        """Test inner product of state with itself."""
        psi = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
        result = psi.bra() * psi
        self.assertAlmostEqual(result, 1)
    
    def test_inner_product_complex(self):
        """Test inner product with complex states."""
        psi = Ket([1, 1j])
        phi = Ket([1j, 1])
        result = psi.bra() * phi
        expected = np.vdot([1, 1j], [1j, 1])
        self.assertAlmostEqual(result, expected)
    
    def test_inner_product_symmetry(self):
        """Test inner product conjugate symmetry."""
        psi = Ket([1, 1j])
        phi = Ket([1j, 1])
        result1 = psi.bra() * phi
        result2 = phi.bra() * psi
        self.assertAlmostEqual(result1, np.conj(result2))


class TestOuterProduct(unittest.TestCase):
    """Test cases for outer product operations."""
    
    def test_outer_product_shape(self):
        """Test outer product produces correct matrix shape."""
        psi = Ket([1, 0])
        phi = Ket([0, 1])
        op = outer_product(psi, phi)
        self.assertEqual(op.matrix.shape, (2, 2))
    
    def test_projection_operator_idempotent(self):
        """Test that projection operator is idempotent (P² = P)."""
        psi = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
        P = projection_operator(psi)
        P_squared = P * P
        np.testing.assert_array_almost_equal(P.matrix, P_squared.matrix)
    
    def test_projection_operator_hermitian(self):
        """Test that projection operator is Hermitian."""
        psi = Ket([1/np.sqrt(2), 1j/np.sqrt(2)])
        P = projection_operator(psi)
        self.assertTrue(P.is_hermitian())


class TestOperator(unittest.TestCase):
    """Test cases for Operator class."""
    
    def test_operator_creation(self):
        """Test operator creation."""
        A = Operator([[1, 0], [0, -1]])
        self.assertEqual(A.matrix.shape, (2, 2))
    
    def test_operator_hermitian_check(self):
        """Test Hermitian operator detection."""
        sigma_z = Operator([[1, 0], [0, -1]])
        self.assertTrue(sigma_z.is_hermitian())
        
        not_hermitian = Operator([[1, 1j], [0, -1]])
        self.assertFalse(not_hermitian.is_hermitian())
    
    def test_operator_unitary_check(self):
        """Test unitary operator detection."""
        hadamard = Operator([[1, 1], [1, -1]]) / np.sqrt(2)
        self.assertTrue(hadamard.is_unitary())
    
    def test_operator_on_ket(self):
        """Test operator acting on ket."""
        sigma_x = Operator([[0, 1], [1, 0]])
        psi = Ket([1, 0])
        result = sigma_x * psi
        np.testing.assert_array_equal(result.state, [0, 1])
    
    def test_operator_multiplication(self):
        """Test operator multiplication."""
        A = Operator([[1, 0], [0, 2]])
        B = Operator([[2, 0], [0, 3]])
        C = A * B
        expected = np.array([[2, 0], [0, 6]])
        np.testing.assert_array_equal(C.matrix, expected)
    
    def test_operator_addition(self):
        """Test operator addition."""
        A = Operator([[1, 0], [0, 1]])
        B = Operator([[2, 0], [0, 2]])
        C = A + B
        expected = np.array([[3, 0], [0, 3]])
        np.testing.assert_array_equal(C.matrix, expected)
    
    def test_commutator(self):
        """Test commutator calculation."""
        sigma_x = PauliMatrices.sigma_x()
        sigma_y = PauliMatrices.sigma_y()
        comm = sigma_x.commutator(sigma_y)
        
        expected = 2j * PauliMatrices.sigma_z().matrix
        np.testing.assert_array_almost_equal(comm.matrix, expected)
    
    def test_anticommutator(self):
        """Test anticommutator calculation."""
        sigma_x = PauliMatrices.sigma_x()
        sigma_y = PauliMatrices.sigma_y()
        anticomm = sigma_x.anticommutator(sigma_y)
        
        expected = np.zeros((2, 2))
        np.testing.assert_array_almost_equal(anticomm.matrix, expected)


class TestExpectationValue(unittest.TestCase):
    """Test cases for expectation value calculations."""
    
    def test_expectation_value_eigenstate(self):
        """Test expectation value in eigenstate."""
        sigma_z = PauliMatrices.sigma_z()
        spin_up = QuantumStates.spin_up()
        
        expectation = sigma_z.expectation_value(spin_up)
        self.assertAlmostEqual(expectation, 1)
    
    def test_expectation_value_superposition(self):
        """Test expectation value in superposition state."""
        sigma_x = PauliMatrices.sigma_x()
        plus_state = QuantumStates.plus_state()
        
        expectation = sigma_x.expectation_value(plus_state)
        self.assertAlmostEqual(expectation, 1)
    
    def test_expectation_value_real(self):
        """Test that expectation value of Hermitian operator is real."""
        sigma_y = PauliMatrices.sigma_y()
        psi = Ket([1/np.sqrt(2), 1j/np.sqrt(2)])
        
        expectation = sigma_y.expectation_value(psi)
        self.assertAlmostEqual(expectation.imag, 0)


class TestEigenvalues(unittest.TestCase):
    """Test cases for eigenvalue problems."""
    
    def test_pauli_z_eigenvalues(self):
        """Test eigenvalues of Pauli-Z."""
        sigma_z = PauliMatrices.sigma_z()
        eigenvals, eigenvecs = sigma_z.eigenvalues_eigenvectors()
        
        np.testing.assert_array_almost_equal(sorted(eigenvals), [-1, 1])
    
    def test_eigenvector_normalization(self):
        """Test that eigenvectors are normalized."""
        sigma_x = PauliMatrices.sigma_x()
        eigenvals, eigenvecs = sigma_x.eigenvalues_eigenvectors()
        
        for vec in eigenvecs:
            self.assertTrue(vec.is_normalized())
    
    def test_eigenvalue_equation(self):
        """Test that eigenvectors satisfy eigenvalue equation."""
        sigma_y = PauliMatrices.sigma_y()
        eigenvals, eigenvecs = sigma_y.eigenvalues_eigenvectors()
        
        for eigenval, eigenvec in zip(eigenvals, eigenvecs):
            result = sigma_y * eigenvec
            expected = eigenval * eigenvec
            np.testing.assert_array_almost_equal(result.state, expected.state)


class TestQuantumStates(unittest.TestCase):
    """Test cases for predefined quantum states."""
    
    def test_spin_states_orthogonal(self):
        """Test that spin up and down are orthogonal."""
        up = QuantumStates.spin_up()
        down = QuantumStates.spin_down()
        
        inner_prod = up.bra() * down
        self.assertAlmostEqual(inner_prod, 0)
    
    def test_plus_minus_orthogonal(self):
        """Test that plus and minus states are orthogonal."""
        plus = QuantumStates.plus_state()
        minus = QuantumStates.minus_state()
        
        inner_prod = plus.bra() * minus
        self.assertAlmostEqual(inner_prod, 0)
    
    def test_states_normalized(self):
        """Test that all predefined states are normalized."""
        states = [
            QuantumStates.spin_up(),
            QuantumStates.spin_down(),
            QuantumStates.plus_state(),
            QuantumStates.minus_state(),
            QuantumStates.right_circular(),
            QuantumStates.left_circular()
        ]
        
        for state in states:
            self.assertTrue(state.is_normalized())


class TestPauliMatrices(unittest.TestCase):
    """Test cases for Pauli matrices."""
    
    def test_pauli_hermitian(self):
        """Test that Pauli matrices are Hermitian."""
        pauli_ops = [
            PauliMatrices.sigma_x(),
            PauliMatrices.sigma_y(),
            PauliMatrices.sigma_z()
        ]
        
        for op in pauli_ops:
            self.assertTrue(op.is_hermitian())
    
    def test_pauli_square_identity(self):
        """Test that σᵢ² = I for all Pauli matrices."""
        pauli_ops = [
            PauliMatrices.sigma_x(),
            PauliMatrices.sigma_y(),
            PauliMatrices.sigma_z()
        ]
        identity = PauliMatrices.identity()
        
        for op in pauli_ops:
            op_squared = op * op
            np.testing.assert_array_almost_equal(op_squared.matrix, identity.matrix)
    
    def test_pauli_eigenvalues(self):
        """Test that Pauli matrices have eigenvalues ±1."""
        pauli_ops = [
            PauliMatrices.sigma_x(),
            PauliMatrices.sigma_y(),
            PauliMatrices.sigma_z()
        ]
        
        for op in pauli_ops:
            eigenvals, _ = op.eigenvalues_eigenvectors()
            np.testing.assert_array_almost_equal(sorted(eigenvals), [-1, 1])
    
    def test_hadamard_unitary(self):
        """Test that Hadamard gate is unitary."""
        H = PauliMatrices.hadamard()
        self.assertTrue(H.is_unitary())


class TestBlochSphere(unittest.TestCase):
    """Test cases for Bloch sphere operations."""
    
    def test_spin_up_bloch_vector(self):
        """Test Bloch vector for spin-up state."""
        up = QuantumStates.spin_up()
        bloch_vec = BlochSphere.state_to_bloch_vector(up)
        
        expected = np.array([0, 0, 1])
        np.testing.assert_array_almost_equal(bloch_vec, expected)
    
    def test_spin_down_bloch_vector(self):
        """Test Bloch vector for spin-down state."""
        down = QuantumStates.spin_down()
        bloch_vec = BlochSphere.state_to_bloch_vector(down)
        
        expected = np.array([0, 0, -1])
        np.testing.assert_array_almost_equal(bloch_vec, expected)
    
    def test_plus_state_bloch_vector(self):
        """Test Bloch vector for plus state."""
        plus = QuantumStates.plus_state()
        bloch_vec = BlochSphere.state_to_bloch_vector(plus)
        
        expected = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(bloch_vec, expected)
    
    def test_bloch_vector_norm(self):
        """Test that Bloch vectors have norm ≤ 1."""
        states = [
            QuantumStates.spin_up(),
            QuantumStates.plus_state(),
            QuantumStates.right_circular()
        ]
        
        for state in states:
            bloch_vec = BlochSphere.state_to_bloch_vector(state)
            norm = np.linalg.norm(bloch_vec)
            self.assertLessEqual(norm, 1 + 1e-10)
    
    def test_bloch_roundtrip(self):
        """Test conversion from state to Bloch vector and back."""
        original_state = QuantumStates.right_circular()
        bloch_vec = BlochSphere.state_to_bloch_vector(original_state)
        reconstructed_state = BlochSphere.bloch_vector_to_state(bloch_vec)
        
        inner_prod = original_state.bra() * reconstructed_state
        self.assertAlmostEqual(abs(inner_prod), 1)


class TestAdvancedOperations(unittest.TestCase):
    """Test cases for advanced quantum operations."""
    
    def test_uncertainty_relation(self):
        """Test Heisenberg uncertainty relation for Pauli matrices."""
        sigma_x = PauliMatrices.sigma_x()
        sigma_y = PauliMatrices.sigma_y()
        
        psi = Ket([1/np.sqrt(3), np.sqrt(2/3)])
        
        exp_x = sigma_x.expectation_value(psi).real
        exp_y = sigma_y.expectation_value(psi).real
        
        exp_x2 = (sigma_x * sigma_x).expectation_value(psi).real
        exp_y2 = (sigma_y * sigma_y).expectation_value(psi).real
        
        delta_x = np.sqrt(exp_x2 - exp_x**2)
        delta_y = np.sqrt(exp_y2 - exp_y**2)
        
        comm = sigma_x.commutator(sigma_y)
        exp_comm = abs(comm.expectation_value(psi))
        
        self.assertGreaterEqual(delta_x * delta_y, exp_comm / 2 - 1e-10)
    
    def test_completeness_relation(self):
        """Test completeness relation Σᵢ |ψᵢ⟩⟨ψᵢ| = I."""
        up = QuantumStates.spin_up()
        down = QuantumStates.spin_down()
        
        P_up = projection_operator(up)
        P_down = projection_operator(down)
        
        completeness = P_up + P_down
        identity = PauliMatrices.identity()
        
        np.testing.assert_array_almost_equal(completeness.matrix, identity.matrix)


if __name__ == '__main__':
    unittest.main()
