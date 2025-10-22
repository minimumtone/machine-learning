"""
Comprehensive examples demonstrating bra-ket notation applications.

This module contains detailed examples for educational purposes, covering:
- Basic quantum state operations
- Spin-1/2 systems
- Pauli matrix calculations
- Measurement and expectation values
- Time evolution
- Multi-particle systems
"""

import numpy as np
import matplotlib.pyplot as plt
from braket_notation import (
    Ket, Bra, Operator,
    outer_product, projection_operator,
    QuantumStates, PauliMatrices, BlochSphere
)


def example_1_basic_states():
    """
    Example 1: Basic quantum states and normalization.
    
    Demonstrates:
    - Creating quantum states
    - Normalization
    - Inner products
    - Orthogonality
    """
    print("=" * 70)
    print("Example 1: Basic Quantum States")
    print("=" * 70)
    
    spin_up = QuantumStates.spin_up()
    spin_down = QuantumStates.spin_down()
    
    print("\n1.1 Spin eigenstates:")
    print(f"|↑⟩ = {spin_up.state}")
    print(f"|↓⟩ = {spin_down.state}")
    
    print(f"\n⟨↑|↑⟩ = {spin_up.bra() * spin_up}")
    print(f"⟨↓|↓⟩ = {spin_down.bra() * spin_down}")
    
    print(f"⟨↑|↓⟩ = {spin_up.bra() * spin_down}")
    
    plus_state = QuantumStates.plus_state()
    minus_state = QuantumStates.minus_state()
    
    print("\n1.2 Superposition states:")
    print(f"|+⟩ = (|↑⟩ + |↓⟩)/√2 = {plus_state.state}")
    print(f"|-⟩ = (|↑⟩ - |↓⟩)/√2 = {minus_state.state}")
    
    print(f"\n⟨+|+⟩ = {plus_state.bra() * plus_state}")
    print(f"⟨+|-⟩ = {plus_state.bra() * minus_state}")
    
    print("\n1.3 State decomposition:")
    coeff_up = spin_up.bra() * plus_state
    coeff_down = spin_down.bra() * plus_state
    print(f"|+⟩ = {coeff_up}|↑⟩ + {coeff_down}|↓⟩")
    print(f"Probability of measuring |↑⟩: |⟨↑|+⟩|² = {abs(coeff_up)**2}")
    print(f"Probability of measuring |↓⟩: |⟨↓|+⟩|² = {abs(coeff_down)**2}")
    
    print("\n" + "=" * 70 + "\n")


def example_2_pauli_matrices():
    """
    Example 2: Pauli matrices and their properties.
    
    Demonstrates:
    - Pauli matrix definitions
    - Hermiticity
    - Eigenvalues and eigenvectors
    - Commutation relations
    """
    print("=" * 70)
    print("Example 2: Pauli Matrices")
    print("=" * 70)
    
    sigma_x = PauliMatrices.sigma_x()
    sigma_y = PauliMatrices.sigma_y()
    sigma_z = PauliMatrices.sigma_z()
    identity = PauliMatrices.identity()
    
    print("\n2.1 Pauli matrix definitions:")
    print(f"\nσₓ =\n{sigma_x.matrix}")
    print(f"\nσᵧ =\n{sigma_y.matrix}")
    print(f"\nσᵤ =\n{sigma_z.matrix}")
    
    print("\n2.2 Properties:")
    print(f"σₓ is Hermitian: {sigma_x.is_hermitian()}")
    print(f"σᵧ is Hermitian: {sigma_y.is_hermitian()}")
    print(f"σᵤ is Hermitian: {sigma_z.is_hermitian()}")
    
    print("\n2.3 Verify σᵢ² = I:")
    sigma_x_squared = sigma_x * sigma_x
    print(f"σₓ² =\n{sigma_x_squared.matrix}")
    print(f"σₓ² = I: {np.allclose(sigma_x_squared.matrix, identity.matrix)}")
    
    print("\n2.4 Eigenvalues and eigenvectors:")
    for name, sigma in [("σₓ", sigma_x), ("σᵧ", sigma_y), ("σᵤ", sigma_z)]:
        eigenvals, eigenvecs = sigma.eigenvalues_eigenvectors()
        print(f"\n{name}:")
        for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs)):
            print(f"  λ_{i} = {val:+.3f}, |ψ_{i}⟩ = {vec.state}")
    
    print("\n2.5 Commutation relations:")
    comm_xy = sigma_x.commutator(sigma_y)
    print(f"[σₓ, σᵧ] =\n{comm_xy.matrix}")
    print(f"2iσᵤ =\n{(2j * sigma_z).matrix}")
    print(f"[σₓ, σᵧ] = 2iσᵤ: {np.allclose(comm_xy.matrix, (2j * sigma_z).matrix)}")
    
    print("\n2.6 Anticommutation relations:")
    anticomm_xy = sigma_x.anticommutator(sigma_y)
    print(f"{{σₓ, σᵧ}} =\n{anticomm_xy.matrix}")
    print(f"{{σₓ, σᵧ}} = 0: {np.allclose(anticomm_xy.matrix, np.zeros((2, 2)))}")
    
    print("\n" + "=" * 70 + "\n")


def example_3_expectation_values():
    """
    Example 3: Expectation values and measurements.
    
    Demonstrates:
    - Computing expectation values
    - Physical interpretation
    - Measurement probabilities
    """
    print("=" * 70)
    print("Example 3: Expectation Values and Measurements")
    print("=" * 70)
    
    sigma_x = PauliMatrices.sigma_x()
    sigma_y = PauliMatrices.sigma_y()
    sigma_z = PauliMatrices.sigma_z()
    
    states = [
        ("Spin up |↑⟩", QuantumStates.spin_up()),
        ("Spin down |↓⟩", QuantumStates.spin_down()),
        ("Plus state |+⟩", QuantumStates.plus_state()),
        ("Minus state |-⟩", QuantumStates.minus_state()),
        ("Right circular |R⟩", QuantumStates.right_circular()),
        ("Left circular |L⟩", QuantumStates.left_circular())
    ]
    
    print("\n3.1 Expectation values for different states:\n")
    print(f"{'State':<25} {'⟨σₓ⟩':>10} {'⟨σᵧ⟩':>10} {'⟨σᵤ⟩':>10}")
    print("-" * 60)
    
    for name, state in states:
        exp_x = sigma_x.expectation_value(state).real
        exp_y = sigma_y.expectation_value(state).real
        exp_z = sigma_z.expectation_value(state).real
        print(f"{name:<25} {exp_x:>10.4f} {exp_y:>10.4f} {exp_z:>10.4f}")
    
    print("\n3.2 Detailed measurement analysis for |+⟩ state:")
    plus_state = QuantumStates.plus_state()
    
    print("\nMeasurement in σᵤ basis:")
    spin_up = QuantumStates.spin_up()
    spin_down = QuantumStates.spin_down()
    
    prob_up = abs(spin_up.bra() * plus_state)**2
    prob_down = abs(spin_down.bra() * plus_state)**2
    
    print(f"P(↑) = |⟨↑|+⟩|² = {prob_up:.4f}")
    print(f"P(↓) = |⟨↓|+⟩|² = {prob_down:.4f}")
    print(f"Sum of probabilities: {prob_up + prob_down:.4f}")
    
    print("\nExpectation value:")
    exp_z = sigma_z.expectation_value(plus_state).real
    print(f"⟨σᵤ⟩ = (+1)×P(↑) + (-1)×P(↓) = {exp_z:.4f}")
    
    print("\n3.3 Verify expectation values are real for Hermitian operators:")
    general_state = Ket([0.6, 0.8j])
    general_state = general_state.normalize()
    
    exp_x = sigma_x.expectation_value(general_state)
    exp_y = sigma_y.expectation_value(general_state)
    exp_z = sigma_z.expectation_value(general_state)
    
    print(f"⟨σₓ⟩ = {exp_x} (imaginary part: {exp_x.imag:.2e})")
    print(f"⟨σᵧ⟩ = {exp_y} (imaginary part: {exp_y.imag:.2e})")
    print(f"⟨σᵤ⟩ = {exp_z} (imaginary part: {exp_z.imag:.2e})")
    
    print("\n" + "=" * 70 + "\n")


def example_4_projection_operators():
    """
    Example 4: Projection operators and measurement.
    
    Demonstrates:
    - Creating projection operators
    - Idempotency property
    - Measurement postulate
    """
    print("=" * 70)
    print("Example 4: Projection Operators")
    print("=" * 70)
    
    spin_up = QuantumStates.spin_up()
    spin_down = QuantumStates.spin_down()
    
    P_up = projection_operator(spin_up)
    P_down = projection_operator(spin_down)
    
    print("\n4.1 Projection operators:")
    print(f"\nP₊ = |↑⟩⟨↑| =\n{P_up.matrix}")
    print(f"\nP₋ = |↓⟩⟨↓| =\n{P_down.matrix}")
    
    print("\n4.2 Verify idempotency (P² = P):")
    P_up_squared = P_up * P_up
    print(f"P₊² =\n{P_up_squared.matrix}")
    print(f"P₊² = P₊: {np.allclose(P_up.matrix, P_up_squared.matrix)}")
    
    print("\n4.3 Verify orthogonality (P₊P₋ = 0):")
    P_product = P_up * P_down
    print(f"P₊P₋ =\n{P_product.matrix}")
    print(f"P₊P₋ = 0: {np.allclose(P_product.matrix, np.zeros((2, 2)))}")
    
    print("\n4.4 Verify completeness (P₊ + P₋ = I):")
    P_sum = P_up + P_down
    identity = PauliMatrices.identity()
    print(f"P₊ + P₋ =\n{P_sum.matrix}")
    print(f"P₊ + P₋ = I: {np.allclose(P_sum.matrix, identity.matrix)}")
    
    print("\n4.5 Apply projection to |+⟩ state:")
    plus_state = QuantumStates.plus_state()
    
    projected_up = P_up * plus_state
    print(f"\nP₊|+⟩ = {projected_up.state}")
    print(f"Norm: {np.sqrt(np.vdot(projected_up.state, projected_up.state).real):.4f}")
    
    prob_up = np.vdot(projected_up.state, projected_up.state).real
    print(f"Probability of measuring |↑⟩: {prob_up:.4f}")
    
    print("\n" + "=" * 70 + "\n")


def example_5_bloch_sphere():
    """
    Example 5: Bloch sphere representation.
    
    Demonstrates:
    - Converting states to Bloch vectors
    - Visualizing states on Bloch sphere
    - Geometric interpretation
    """
    print("=" * 70)
    print("Example 5: Bloch Sphere Representation")
    print("=" * 70)
    
    states = [
        ("Spin up |↑⟩", QuantumStates.spin_up()),
        ("Spin down |↓⟩", QuantumStates.spin_down()),
        ("Plus |+⟩", QuantumStates.plus_state()),
        ("Minus |-⟩", QuantumStates.minus_state()),
        ("Right |R⟩", QuantumStates.right_circular()),
        ("Left |L⟩", QuantumStates.left_circular())
    ]
    
    print("\n5.1 Bloch vectors for common states:\n")
    print(f"{'State':<20} {'Bloch Vector (x, y, z)':<30}")
    print("-" * 50)
    
    for name, state in states:
        bloch_vec = BlochSphere.state_to_bloch_vector(state)
        print(f"{name:<20} ({bloch_vec[0]:>6.3f}, {bloch_vec[1]:>6.3f}, {bloch_vec[2]:>6.3f})")
    
    print("\n5.2 Verify Bloch vector properties:")
    for name, state in states:
        bloch_vec = BlochSphere.state_to_bloch_vector(state)
        norm = np.linalg.norm(bloch_vec)
        print(f"{name:<20} norm = {norm:.6f}")
    
    print("\n5.3 Creating Bloch sphere visualization...")
    state_list = [state for _, state in states]
    labels = [name for name, _ in states]
    
    fig = BlochSphere.plot_bloch_sphere(state_list, labels, 
                                        "Bloch Sphere: Common Quantum States")
    plt.savefig('/home/ubuntu/repos/machine-learning/bloch_sphere_example.png', 
                dpi=150, bbox_inches='tight')
    print("Saved to: bloch_sphere_example.png")
    plt.close()
    
    print("\n5.4 Roundtrip conversion (state → Bloch → state):")
    test_state = QuantumStates.right_circular()
    print(f"Original state: {test_state.state}")
    
    bloch_vec = BlochSphere.state_to_bloch_vector(test_state)
    print(f"Bloch vector: {bloch_vec}")
    
    reconstructed = BlochSphere.bloch_vector_to_state(bloch_vec)
    print(f"Reconstructed state: {reconstructed.state}")
    
    fidelity = abs(test_state.bra() * reconstructed)
    print(f"Fidelity: {fidelity:.6f}")
    
    print("\n" + "=" * 70 + "\n")


def example_6_time_evolution():
    """
    Example 6: Time evolution under Hamiltonian.
    
    Demonstrates:
    - Time evolution operator
    - Unitary evolution
    - Precession on Bloch sphere
    """
    print("=" * 70)
    print("Example 6: Time Evolution")
    print("=" * 70)
    
    omega = 1.0  # Angular frequency
    sigma_z = PauliMatrices.sigma_z()
    H = omega * sigma_z
    
    print(f"\n6.1 Hamiltonian H = ω σᵤ with ω = {omega}:")
    print(f"\nH =\n{H.matrix}")
    
    initial_state = QuantumStates.plus_state()
    print(f"\n6.2 Initial state |ψ(0)⟩ = |+⟩:")
    print(f"State vector: {initial_state.state}")
    
    bloch_initial = BlochSphere.state_to_bloch_vector(initial_state)
    print(f"Bloch vector: {bloch_initial}")
    
    times = np.linspace(0, 2*np.pi/omega, 50)
    states_evolved = []
    bloch_vectors = []
    
    print("\n6.3 Time evolution:")
    for t in [0, np.pi/(4*omega), np.pi/(2*omega), np.pi/omega, 2*np.pi/omega]:
        U_t = Operator(np.linalg.matrix_power(
            np.eye(2) + (-1j * H.matrix * t / 100), 100
        ))
        from scipy.linalg import expm
        U_t = Operator(expm(-1j * H.matrix * t))
        
        state_t = U_t * initial_state
        bloch_t = BlochSphere.state_to_bloch_vector(state_t)
        
        print(f"\nt = {t:.4f}:")
        print(f"  State: {state_t.state}")
        print(f"  Bloch: ({bloch_t[0]:.4f}, {bloch_t[1]:.4f}, {bloch_t[2]:.4f})")
    
    print("\n6.4 Creating time evolution visualization...")
    for t in times:
        from scipy.linalg import expm
        U_t = Operator(expm(-1j * H.matrix * t))
        state_t = U_t * initial_state
        states_evolved.append(state_t)
        bloch_vectors.append(BlochSphere.state_to_bloch_vector(state_t))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')
    
    bloch_array = np.array(bloch_vectors)
    ax.plot(bloch_array[:, 0], bloch_array[:, 1], bloch_array[:, 2], 
            'r-', linewidth=2, label='Trajectory')
    
    ax.scatter([bloch_array[0, 0]], [bloch_array[0, 1]], [bloch_array[0, 2]], 
               color='green', s=100, label='Initial')
    ax.scatter([bloch_array[-1, 0]], [bloch_array[-1, 1]], [bloch_array[-1, 2]], 
               color='red', s=100, label='Final')
    
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Time Evolution on Bloch Sphere')
    ax.legend()
    
    plt.savefig('/home/ubuntu/repos/machine-learning/time_evolution_bloch.png', 
                dpi=150, bbox_inches='tight')
    print("Saved to: time_evolution_bloch.png")
    plt.close()
    
    print("\n" + "=" * 70 + "\n")


def example_7_tensor_products():
    """
    Example 7: Multi-particle systems and tensor products.
    
    Demonstrates:
    - Tensor product of states
    - Entangled states
    - Bell states
    """
    print("=" * 70)
    print("Example 7: Multi-Particle Systems")
    print("=" * 70)
    
    spin_up = QuantumStates.spin_up()
    spin_down = QuantumStates.spin_down()
    
    print("\n7.1 Product states:")
    
    up_up = spin_up.tensor_product(spin_up)
    print(f"\n|↑↑⟩ = |↑⟩ ⊗ |↑⟩ = {up_up.state}")
    
    up_down = spin_up.tensor_product(spin_down)
    print(f"|↑↓⟩ = |↑⟩ ⊗ |↓⟩ = {up_down.state}")
    
    down_up = spin_down.tensor_product(spin_up)
    print(f"|↓↑⟩ = |↓⟩ ⊗ |↑⟩ = {down_up.state}")
    
    down_down = spin_down.tensor_product(spin_down)
    print(f"|↓↓⟩ = |↓⟩ ⊗ |↓⟩ = {down_down.state}")
    
    print("\n7.2 Bell states (maximally entangled):")
    
    bell_phi_plus = (up_up + down_down) / np.sqrt(2)
    print(f"\n|Φ⁺⟩ = (|↑↑⟩ + |↓↓⟩)/√2 = {bell_phi_plus.state}")
    
    bell_phi_minus = (up_up - down_down) / np.sqrt(2)
    print(f"|Φ⁻⟩ = (|↑↑⟩ - |↓↓⟩)/√2 = {bell_phi_minus.state}")
    
    bell_psi_plus = (up_down + down_up) / np.sqrt(2)
    print(f"|Ψ⁺⟩ = (|↑↓⟩ + |↓↑⟩)/√2 = {bell_psi_plus.state}")
    
    bell_psi_minus = (up_down - down_up) / np.sqrt(2)
    print(f"|Ψ⁻⟩ = (|↑↓⟩ - |↓↑⟩)/√2 = {bell_psi_minus.state}")
    
    print("\n7.3 Verify Bell states are normalized and orthogonal:")
    bell_states = [
        ("Φ⁺", bell_phi_plus),
        ("Φ⁻", bell_phi_minus),
        ("Ψ⁺", bell_psi_plus),
        ("Ψ⁻", bell_psi_minus)
    ]
    
    for name, state in bell_states:
        norm = np.sqrt(np.vdot(state.state, state.state).real)
        print(f"⟨{name}|{name}⟩ = {norm:.6f}")
    
    print("\nOrthogonality:")
    for i, (name1, state1) in enumerate(bell_states):
        for name2, state2 in bell_states[i+1:]:
            inner_prod = state1.bra() * state2
            print(f"⟨{name1}|{name2}⟩ = {inner_prod:.6f}")
    
    print("\n" + "=" * 70 + "\n")


def example_8_uncertainty_principle():
    """
    Example 8: Heisenberg uncertainty principle.
    
    Demonstrates:
    - Uncertainty calculation
    - Complementary observables
    - Minimum uncertainty states
    """
    print("=" * 70)
    print("Example 8: Uncertainty Principle")
    print("=" * 70)
    
    sigma_x = PauliMatrices.sigma_x()
    sigma_y = PauliMatrices.sigma_y()
    sigma_z = PauliMatrices.sigma_z()
    
    def calculate_uncertainty(operator, state):
        """Calculate uncertainty Δ A = √(⟨A²⟩ - ⟨A⟩²)"""
        exp_A = operator.expectation_value(state).real
        A_squared = operator * operator
        exp_A2 = A_squared.expectation_value(state).real
        delta_A = np.sqrt(exp_A2 - exp_A**2)
        return delta_A
    
    print("\n8.1 Uncertainty for different states:\n")
    
    states = [
        ("Spin up |↑⟩", QuantumStates.spin_up()),
        ("Plus state |+⟩", QuantumStates.plus_state()),
        ("General state", Ket([0.6, 0.8]).normalize())
    ]
    
    print(f"{'State':<20} {'Δσₓ':>10} {'Δσᵧ':>10} {'Δσᵤ':>10} {'Δσₓ·Δσᵧ':>12}")
    print("-" * 70)
    
    for name, state in states:
        delta_x = calculate_uncertainty(sigma_x, state)
        delta_y = calculate_uncertainty(sigma_y, state)
        delta_z = calculate_uncertainty(sigma_z, state)
        product_xy = delta_x * delta_y
        
        print(f"{name:<20} {delta_x:>10.4f} {delta_y:>10.4f} {delta_z:>10.4f} {product_xy:>12.4f}")
    
    print("\n8.2 Uncertainty relation:")
    print("For non-commuting observables A and B:")
    print("Δ A · Δ B ≥ |⟨[A,B]⟩| / 2")
    
    state = Ket([0.6, 0.8]).normalize()
    delta_x = calculate_uncertainty(sigma_x, state)
    delta_y = calculate_uncertainty(sigma_y, state)
    
    comm = sigma_x.commutator(sigma_y)
    exp_comm = abs(comm.expectation_value(state))
    lower_bound = exp_comm / 2
    
    print(f"\nFor state {state.state}:")
    print(f"Δσₓ · Δσᵧ = {delta_x * delta_y:.4f}")
    print(f"|⟨[σₓ,σᵧ]⟩| / 2 = {lower_bound:.4f}")
    print(f"Uncertainty relation satisfied: {delta_x * delta_y >= lower_bound - 1e-10}")
    
    print("\n" + "=" * 70 + "\n")


def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "=" * 70)
    print(" " * 15 + "BRA-KET NOTATION EXAMPLES")
    print(" " * 10 + "Comprehensive Quantum Mechanics Tutorial")
    print("=" * 70 + "\n")
    
    example_1_basic_states()
    example_2_pauli_matrices()
    example_3_expectation_values()
    example_4_projection_operators()
    example_5_bloch_sphere()
    example_6_time_evolution()
    example_7_tensor_products()
    example_8_uncertainty_principle()
    
    print("=" * 70)
    print(" " * 20 + "ALL EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
