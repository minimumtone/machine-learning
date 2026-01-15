"""
Extended Hubbard Hamiltonian Construction Module

This module constructs the extended Hubbard Hamiltonian from DFT+U+V data:
H = sum_pq t_pq a^dag_p,sigma a_q,sigma + sum_p U_p n_p,up n_p,down + sum_pq V_pq n_p,sigma n_q,tau

The Hamiltonian is built from:
1. Hopping matrix (t_pq) from tight-binding projection
2. On-site Hubbard U parameters
3. Inter-site Hubbard V parameters
"""

import numpy as np
from typing import Dict, Tuple, Any


def make_two_body_tensor(hopping_matrix: np.ndarray, 
                         hubbard_params: Dict[Tuple[int, int], Dict[str, Any]]) -> np.ndarray:
    """
    Construct the two-body tensor from Hubbard parameters.
    
    The two-body tensor encodes the electron-electron interactions:
    - On-site (U): two_body_tensor[i,i,i,i] = 2*U
    - Inter-site (V): two_body_tensor[i,i,j,j] = 2*V
    
    Args:
        hopping_matrix: One-body hopping matrix from tight-binding projection
        hubbard_params: Dictionary of Hubbard parameters with structure:
            {(species_i, species_j): {'V': value, 'index_i': [...], 'index_j': [...]}}
    
    Returns:
        two_body_tensor: 4D tensor of shape (n_orb, n_orb, n_orb, n_orb)
    """
    num_orbitals = hopping_matrix.shape[0]
    two_body_tensor = np.zeros((num_orbitals, num_orbitals, num_orbitals, num_orbitals))
    
    for k, v_dict in hubbard_params.items():
        if k[0] == k[1]:
            # On-site interaction (U)
            for i in v_dict["index_i"]:
                two_body_tensor[i, i, i, i] = 2 * v_dict["V"]
        else:
            # Inter-site interaction (V)
            for i in v_dict["index_i"]:
                for j in v_dict["index_j"]:
                    two_body_tensor[i, i, j, j] = 2 * v_dict["V"]
                    two_body_tensor[j, j, i, i] = 2 * v_dict["V"]
    
    return two_body_tensor


def rotate_tensors(hopping_matrix: np.ndarray, 
                   two_body_tensor: np.ndarray, 
                   basis_rotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate one-body and two-body tensors to molecular orbital basis.
    
    Args:
        hopping_matrix: One-body hopping matrix in atomic orbital basis
        two_body_tensor: Two-body tensor in atomic orbital basis
        basis_rotation: Unitary transformation matrix (MO coefficients)
    
    Returns:
        hopping_matrix_rot: Rotated one-body matrix
        two_body_tensor_rot: Rotated two-body tensor
    """
    hopping_matrix_rot = basis_rotation.T @ hopping_matrix @ basis_rotation
    two_body_tensor_rot = np.einsum('pqrs,pi,qj,rk,sl->ijkl', 
                                     two_body_tensor, 
                                     basis_rotation, basis_rotation, 
                                     basis_rotation, basis_rotation)
    return hopping_matrix_rot, two_body_tensor_rot


def build_molecular_hamiltonian(hopping_matrix: np.ndarray,
                                two_body_tensor: np.ndarray,
                                mo_coeff: np.ndarray = None):
    """
    Build molecular Hamiltonian using ffsim.
    
    Args:
        hopping_matrix: One-body hopping matrix
        two_body_tensor: Two-body interaction tensor
        mo_coeff: Molecular orbital coefficients (optional, for basis rotation)
    
    Returns:
        hamiltonian: ffsim MolecularHamiltonian object
    """
    try:
        import ffsim
        
        hamiltonian = ffsim.MolecularHamiltonian(
            one_body_tensor=hopping_matrix,
            two_body_tensor=two_body_tensor
        )
        
        if mo_coeff is not None:
            hamiltonian = hamiltonian.rotated(mo_coeff.T)
        
        return hamiltonian
    except ImportError:
        raise ImportError("ffsim is required for Hamiltonian construction. "
                         "Install with: pip install ffsim")
