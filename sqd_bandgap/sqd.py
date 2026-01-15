"""
SQD (Sample-based Quantum Diagonalization) Module

This module implements the SQD algorithm:
1. Process quantum measurement counts
2. Subsample and postselect configurations
3. Perform classical diagonalization in the sampled subspace
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any


def make_hf_bitstring(nelec: Tuple[int, int], num_orbitals: int) -> str:
    """
    Create Hartree-Fock reference bitstring.
    
    Args:
        nelec: Tuple of (n_alpha, n_beta) electrons
        num_orbitals: Number of spatial orbitals
    
    Returns:
        hf_string: Hartree-Fock bitstring
    """
    n_alpha, n_beta = nelec
    alpha_bits = '1' * n_alpha + '0' * (num_orbitals - n_alpha)
    beta_bits = '1' * n_beta + '0' * (num_orbitals - n_beta)
    return alpha_bits + beta_bits


def process_sqd_batch(batch: np.ndarray,
                      hcore: np.ndarray,
                      eri: np.ndarray,
                      nuclear_repulsion_energy: float,
                      open_shell: bool,
                      batch_idx: int,
                      max_davidson_cycles: int = 800,
                      hf_string: str = None) -> Tuple[float, np.ndarray, np.ndarray, Tuple]:
    """
    Process a single SQD batch using PySCF's selected CI solver.
    
    Args:
        batch: Bitstring matrix for this batch
        hcore: One-body Hamiltonian
        eri: Two-electron integrals
        nuclear_repulsion_energy: Nuclear repulsion energy
        open_shell: Whether the system is open shell
        batch_idx: Index of this batch
        max_davidson_cycles: Maximum Davidson iterations
        hf_string: Hartree-Fock reference bitstring
    
    Returns:
        energy: Ground state energy
        occupancies: Orbital occupancies
        ci_coeffs: CI coefficients
        addresses: (alpha_addresses, beta_addresses)
    """
    try:
        import pyscf
        from pyscf import ao2mo
        from pyscf.fci import direct_spin1
        from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs
    except ImportError:
        raise ImportError("pyscf and qiskit-addon-sqd are required.")
    
    # Convert bitstrings to CI strings
    addresses = bitstring_matrix_to_ci_strs(batch, open_shell=open_shell)
    
    num_orbitals = hcore.shape[0]
    n_alpha = batch.shape[1] // 2
    n_beta = batch.shape[1] // 2
    nelec = (n_alpha, n_beta)
    
    # Setup selected CI solver
    myci = pyscf.fci.selected_ci.SelectedCI()
    
    # Absorb one-body into two-body
    h2e_abs = direct_spin1.absorb_h1e(hcore, eri, num_orbitals, nelec, 0.5)
    h2e_abs = ao2mo.restore(1, h2e_abs, num_orbitals)
    
    # Solve in the selected subspace
    try:
        e, ci = myci.kernel(hcore, eri, num_orbitals, nelec, 
                           ci0=None, ecore=nuclear_repulsion_energy,
                           max_cycle=max_davidson_cycles)
    except Exception as ex:
        print(f"Batch {batch_idx} failed: {ex}")
        return np.inf, np.zeros(2 * num_orbitals), None, addresses
    
    # Compute occupancies
    if ci is not None:
        dm1 = myci.make_rdm1(ci, num_orbitals, nelec)
        occupancies = np.diag(dm1)
    else:
        occupancies = np.zeros(2 * num_orbitals)
    
    return e, occupancies, ci, addresses


def run_sqd(counts: Dict[str, int],
            hcore: np.ndarray,
            eri: np.ndarray,
            nelec: Tuple[int, int],
            samples_per_batch: int = 1000,
            n_batches: int = 1,
            recovery_iterations: int = 1,
            rand_seed: int = 469420) -> Dict[str, Any]:
    """
    Run the full SQD algorithm.
    
    Args:
        counts: Quantum measurement counts
        hcore: One-body Hamiltonian
        eri: Two-electron integrals
        nelec: Tuple of (n_alpha, n_beta) electrons
        samples_per_batch: Number of samples per batch
        n_batches: Number of batches
        recovery_iterations: Number of configuration recovery iterations
        rand_seed: Random seed
    
    Returns:
        results: Dictionary containing energies and other results
    """
    try:
        from qiskit_addon_sqd.counts import counts_to_arrays
        from qiskit_addon_sqd.configuration_recovery import recover_configurations
        from qiskit_addon_sqd.subsampling import postselect_and_subsample
    except ImportError:
        raise ImportError("qiskit-addon-sqd is required.")
    
    num_orbitals = hcore.shape[0]
    n_alpha, n_beta = nelec
    open_shell = n_alpha != n_beta
    
    hf_string = make_hf_bitstring(nelec, num_orbitals)
    
    # Convert counts to arrays
    bitstring_matrix_full, probs_arr_full = counts_to_arrays(counts)
    
    e_hist = np.zeros((recovery_iterations, n_batches))
    occupancy_hist = np.zeros((recovery_iterations, n_batches, 2 * num_orbitals))
    occupancies_bitwise = None
    
    for i in range(recovery_iterations):
        print(f"Starting configuration recovery iteration {i}...")
        
        if occupancies_bitwise is None:
            bs_mat_tmp = bitstring_matrix_full
            probs_arr_tmp = probs_arr_full
        else:
            bs_mat_tmp, probs_arr_tmp = recover_configurations(
                bitstring_matrix_full,
                probs_arr_full,
                occupancies_bitwise,
                n_alpha,
                n_beta,
                rand_seed=rand_seed,
            )
        
        # Postselect and subsample
        batches = postselect_and_subsample(
            bs_mat_tmp,
            probs_arr_tmp,
            hamming_right=n_alpha,
            hamming_left=n_beta,
            samples_per_batch=samples_per_batch,
            num_batches=n_batches,
            rand_seed=rand_seed,
        )
        
        # Process each batch
        for j, batch in enumerate(batches):
            e, occs, ci, addresses = process_sqd_batch(
                batch, hcore, eri, 0.0, open_shell, j,
                hf_string=hf_string
            )
            e_hist[i, j] = e
            occupancy_hist[i, j, :len(occs)] = occs
        
        # Update occupancies for next iteration
        best_batch_idx = np.argmin(e_hist[i, :])
        occupancies_bitwise = occupancy_hist[i, best_batch_idx, :]
    
    # Get best result
    best_iter = recovery_iterations - 1
    best_batch = np.argmin(e_hist[best_iter, :])
    best_energy = e_hist[best_iter, best_batch]
    
    return {
        'sqd_energy': best_energy,
        'e_hist': e_hist,
        'occupancy_hist': occupancy_hist,
        'best_batch': best_batch,
        'best_iteration': best_iter
    }
