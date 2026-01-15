"""
Quantum Circuit Module

This module handles the construction of quantum circuits for SQD:
1. LUCJ (Local Unitary Cluster Jastrow) ansatz construction
2. Qiskit circuit generation for simulation
"""

import numpy as np
from typing import Tuple, List, Optional


def make_lucj_circuit(nelec: Tuple[int, int],
                      num_orbitals: int,
                      t2: np.ndarray,
                      layers: int = 1,
                      truncated_lucj: bool = False):
    """
    Construct LUCJ ansatz circuit using ffsim.
    
    The LUCJ ansatz is a hardware-efficient ansatz that captures
    electron correlation through local unitary operations and
    Jastrow factors.
    
    Args:
        nelec: Tuple of (n_alpha, n_beta) electrons
        num_orbitals: Number of spatial orbitals
        t2: CCSD double excitation amplitudes for initialization
        layers: Number of LUCJ layers
        truncated_lucj: Whether to use truncated LUCJ circuit
    
    Returns:
        circ_ffsim: ffsim UCJ operator
    """
    try:
        import ffsim
    except ImportError:
        raise ImportError("ffsim is required. Install with: pip install ffsim")
    
    # Define interaction pairs based on spin configuration
    if nelec[0] == nelec[1]:
        # Closed shell
        alpha_alpha_indices = [(p, p + 1) for p in range(num_orbitals - 1)]
        alpha_beta_indices = [(p, p) for p in range(num_orbitals) if p % 4 == 0]
        interaction_pairs = (alpha_alpha_indices, alpha_beta_indices)
        
        if truncated_lucj:
            n_reps = 2
            circ_ffsim = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                t2, n_reps=n_reps, interaction_pairs=interaction_pairs
            )
            circ_ffsim = ffsim.UCJOpSpinBalanced(
                diag_coulomb_mats=circ_ffsim.diag_coulomb_mats[:-1],
                orbital_rotations=circ_ffsim.orbital_rotations[:-1],
                final_orbital_rotation=circ_ffsim.orbital_rotations[-1]
            )
        else:
            circ_ffsim = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                t2=t2, n_reps=layers, interaction_pairs=interaction_pairs
            )
    else:
        # Open shell
        alpha_alpha_indices = [(p, p + 1) for p in range(num_orbitals - 1)]
        alpha_beta_indices = [(p, p) for p in range(num_orbitals) if p % 4 == 0]
        beta_beta_indices = [(p, p + 1) for p in range(num_orbitals - 1)]
        interaction_pairs = (alpha_alpha_indices, alpha_beta_indices, beta_beta_indices)
        
        if truncated_lucj:
            n_reps = 2
            circ_ffsim = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
                t2, n_reps=n_reps, interaction_pairs=interaction_pairs
            )
            circ_ffsim = ffsim.UCJOpSpinUnbalanced(
                diag_coulomb_mats=circ_ffsim.diag_coulomb_mats[:-1],
                orbital_rotations=circ_ffsim.orbital_rotations[:-1],
                final_orbital_rotation=circ_ffsim.orbital_rotations[-1]
            )
        else:
            circ_ffsim = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
                t2=t2, n_reps=layers, interaction_pairs=interaction_pairs
            )
    
    return circ_ffsim


def make_circuit_qiskit(circ_ffsim,
                        num_orbitals: int,
                        nelec: Tuple[int, int],
                        basis_rotation: np.ndarray,
                        basis: str = 'atomic'):
    """
    Convert ffsim circuit to Qiskit circuit for simulation.
    
    Args:
        circ_ffsim: ffsim UCJ operator
        num_orbitals: Number of spatial orbitals
        nelec: Tuple of (n_alpha, n_beta) electrons
        basis_rotation: Basis rotation matrix (MO coefficients)
        basis: 'atomic' or 'molecular' basis
    
    Returns:
        circ_qiskit: Qiskit QuantumCircuit
    """
    try:
        import ffsim
        from qiskit.circuit import QuantumCircuit, QuantumRegister
    except ImportError:
        raise ImportError("ffsim and qiskit are required.")
    
    qubits = QuantumRegister(2 * num_orbitals, name="q")
    circ_qiskit = QuantumCircuit(qubits)
    
    # Prepare Hartree-Fock state
    circ_qiskit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), qubits)
    
    # Apply LUCJ ansatz
    if nelec[0] == nelec[1]:
        circ_qiskit.append(ffsim.qiskit.UCJOpSpinBalancedJW(circ_ffsim), qubits)
    else:
        circ_qiskit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(circ_ffsim), qubits)
    
    # Apply basis rotation if in atomic basis
    if basis == 'atomic':
        circ_qiskit.append(
            ffsim.qiskit.OrbitalRotationJW(num_orbitals, basis_rotation), 
            qubits
        )
    
    circ_qiskit.measure_all()
    
    return circ_qiskit


def sample_circuit_ffsim(circ_qiskit, shots: int = 1000000, seed: int = 12345) -> dict:
    """
    Sample quantum circuit using ffsim simulator.
    
    This is a noiseless classical simulation of the quantum circuit.
    
    Args:
        circ_qiskit: Qiskit QuantumCircuit
        shots: Number of measurement shots
        seed: Random seed for reproducibility
    
    Returns:
        counts: Dictionary of bitstring counts
    """
    try:
        import ffsim
    except ImportError:
        raise ImportError("ffsim is required. Install with: pip install ffsim")
    
    sampler = ffsim.qiskit.FfsimSampler(default_shots=shots, seed=seed)
    pub = (circ_qiskit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    
    return counts
