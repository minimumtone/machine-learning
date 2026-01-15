"""
Classical Preprocessing Module

This module handles the classical preprocessing steps:
1. Hartree-Fock calculation
2. CCSD calculation for initial ansatz parameters
3. Basis rotation and tensor transformation
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

from .hamiltonian import make_two_body_tensor


def make_pyscf_slater_det(nelec: Tuple[int, int], num_orbitals: int) -> np.ndarray:
    """
    Create occupation vector for Slater determinant.
    
    Args:
        nelec: Tuple of (n_alpha, n_beta) electrons
        num_orbitals: Number of spatial orbitals
    
    Returns:
        occ: Occupation vector
    """
    occ = np.zeros(num_orbitals)
    n_doubly = min(nelec[0], nelec[1])
    n_singly = abs(nelec[0] - nelec[1])
    
    occ[:n_doubly] = 2.0
    if n_singly > 0:
        occ[n_doubly:n_doubly + n_singly] = 1.0
    
    return occ


def run_pyscf_calculations(hopping_matrix: np.ndarray,
                           hubbard_params: Dict[Tuple[int, int], Dict[str, Any]],
                           nelec: Tuple[int, int],
                           compute_fci: bool = False,
                           logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Run PySCF calculations (HF, CCSD, optionally FCI).
    
    Args:
        hopping_matrix: One-body hopping matrix
        hubbard_params: Hubbard parameters dictionary
        nelec: Tuple of (n_alpha, n_beta) electrons
        compute_fci: Whether to compute FCI (expensive for large systems)
        logger: Optional logger for output
    
    Returns:
        results: Dictionary containing energies, MO coefficients, and amplitudes
    """
    try:
        import pyscf
    except ImportError:
        raise ImportError("PySCF is required. Install with: pip install pyscf")
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Starting calculations for nelec: {nelec}")
    
    num_orbitals = hopping_matrix.shape[0]
    two_body_tensor = make_two_body_tensor(hopping_matrix, hubbard_params)
    
    # Setup PySCF molecule object
    mol = pyscf.gto.M(verbose=0)
    mol.nelec = nelec
    mol.incore_anyway = True
    
    # Hartree-Fock calculation
    hf_obj = pyscf.scf.RHF(mol)
    hf_obj.get_hcore = lambda *args: np.real(hopping_matrix)
    hf_obj.get_ovlp = lambda *args: np.eye(num_orbitals)
    hf_obj._eri = two_body_tensor
    
    pyscf_slater_det = make_pyscf_slater_det(nelec, num_orbitals)
    hf_obj.get_occ = lambda *args: pyscf_slater_det
    
    logger.info('Running Hartree-Fock...')
    hf_obj.kernel()
    logger.info(f"HF energy: {hf_obj.e_tot}")
    
    # CCSD calculation
    ccsd_obj = pyscf.cc.CCSD(hf_obj)
    logger.info("Running CCSD...")
    
    max_cycle = ccsd_obj.max_cycle
    while max_cycle > 0:
        try:
            ccsd_obj.kernel()
            logger.info("CCSD converged successfully.")
            break
        except np.linalg.LinAlgError:
            logger.warning(f"LinAlgError with max_cycles={max_cycle}. Retrying...")
            max_cycle -= 1
            ccsd_obj.max_cycle = max_cycle
    
    logger.info(f"CCSD energy: {ccsd_obj.e_tot}")
    
    # CCSD(T) correction
    ccsd_t_correction = ccsd_obj.ccsd_t()
    logger.info(f"CCSD(T) energy: {ccsd_obj.e_tot + ccsd_t_correction}")
    
    results = {
        'hf_energy': hf_obj.e_tot,
        'two_body_tensor': two_body_tensor,
        'ne_ab': nelec,
        'mo_coeff': hf_obj.mo_coeff,
        'ccsd_energy': ccsd_obj.e_tot,
        't1': ccsd_obj.t1,
        't2': ccsd_obj.t2,
        'ccsdt_energy': ccsd_obj.e_tot + ccsd_t_correction,
        'fci_energy': None
    }
    
    # Optional FCI calculation
    if compute_fci:
        logger.info("Running FCI...")
        fci_obj = pyscf.fci.FCI(hf_obj)
        fci_obj.kernel()
        results['fci_energy'] = fci_obj.e_tot
        logger.info(f"FCI energy: {fci_obj.e_tot}")
    
    return results
