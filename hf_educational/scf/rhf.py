"""
Restricted Hartree-Fock (RHF) implementation.

Implements closed-shell RHF with:
- SCF iterations with damping
- DIIS convergence acceleration
- Energy decomposition analysis
- Convergence monitoring
"""

import numpy as np
from typing import Optional, Dict, List
import time

from basis.basis_set import BasisSet
from molecule_io.molecule import Molecule
from integrals import (compute_overlap_matrix, compute_kinetic_matrix,
                         compute_nuclear_attraction_matrix, compute_eri_tensor)
from linalg import symmetric_orthogonalization, solve_generalized_eigenproblem
from .diis import DIIS


class RHF:
    """
    Restricted Hartree-Fock for closed-shell systems.
    
    Solves the Roothaan-Hall equations:
    F C = S C ε
    
    where F = H_core + G[P]
    G = J - (1/2)K
    """
    
    def __init__(self, molecule: Molecule, basis_name: str = 'sto-3g'):
        """
        Initialize RHF calculation.
        
        Args:
            molecule: Molecule object
            basis_name: Basis set name
        """
        if not molecule.is_closed_shell():
            raise ValueError("RHF requires closed-shell molecule (multiplicity=1)")
        
        self.molecule = molecule
        self.basis_name = basis_name
        
        print(f"\n{'='*60}")
        print("Restricted Hartree-Fock (RHF) Calculation")
        print(f"{'='*60}")
        print(self.molecule)
        print(f"\nBasis set: {basis_name.upper()}")
        
        self.basis = BasisSet(molecule, basis_name)
        print(f"Number of basis functions: {self.basis.n_basis}")
        
        self.n_occ = molecule.n_alpha
        print(f"Number of occupied orbitals: {self.n_occ}")
        
        self.S = None
        self.T = None
        self.V = None
        self.H_core = None
        self.ERI = None
        self.X = None
        
        self.P = None
        self.F = None
        self.C = None
        self.eps = None
        
        self.E_total = 0.0
        self.E_elec = 0.0
        self.E_nuc = 0.0
        
        self.converged = False
        self.iteration = 0
        
        self.energy_history = []
        self.convergence_history = []
        
    def compute_integrals(self, schwarz_threshold: float = 1e-10):
        """
        Compute all required integrals.
        
        Args:
            schwarz_threshold: Threshold for Schwarz screening in ERI
        """
        print(f"\n{'='*60}")
        print("Computing Integrals")
        print(f"{'='*60}")
        
        start = time.time()
        
        print("\n1. Overlap matrix S...")
        self.S = compute_overlap_matrix(self.basis)
        print(f"   Computed in {time.time() - start:.2f}s")
        print(f"   Condition number: {np.linalg.cond(self.S):.2e}")
        
        start = time.time()
        print("\n2. Kinetic energy matrix T...")
        self.T = compute_kinetic_matrix(self.basis)
        print(f"   Computed in {time.time() - start:.2f}s")
        
        start = time.time()
        print("\n3. Nuclear attraction matrix V...")
        self.V = compute_nuclear_attraction_matrix(self.basis, self.molecule)
        print(f"   Computed in {time.time() - start:.2f}s")
        
        self.H_core = self.T + self.V
        print("\n4. Core Hamiltonian H = T + V")
        
        self.E_nuc = self.molecule.nuclear_repulsion()
        print(f"\n5. Nuclear repulsion energy: {self.E_nuc:.10f} Eh")
        
        start = time.time()
        print("\n6. Two-electron integrals (ERI)...")
        print(f"   Schwarz threshold: {schwarz_threshold}")
        self.ERI = compute_eri_tensor(self.basis, schwarz_threshold)
        print(f"   Computed in {time.time() - start:.2f}s")
        
        print("\n7. Symmetric orthogonalization X = S^(-1/2)...")
        self.X = symmetric_orthogonalization(self.S)
        
    def initial_guess(self, method: str = 'core'):
        """
        Generate initial density matrix guess.
        
        Args:
            method: 'core' (H_core eigenvectors) or 'zero'
        """
        print(f"\n{'='*60}")
        print(f"Initial Guess: {method}")
        print(f"{'='*60}")
        
        if method == 'core':
            eps, C = solve_generalized_eigenproblem(self.H_core, self.S, self.X)
            C_occ = C[:, :self.n_occ]
            self.P = 2.0 * C_occ @ C_occ.T
            
            print("Initial orbital energies (occupied):")
            for i in range(self.n_occ):
                print(f"  MO {i+1}: {eps[i]:12.6f} Eh")
        
        elif method == 'zero':
            self.P = np.zeros((self.basis.n_basis, self.basis.n_basis))
        
        else:
            raise ValueError(f"Unknown initial guess method: {method}")
        
        print(f"\nInitial density matrix trace: {np.trace(self.P @ self.S):.6f}")
        print(f"(Should be {2 * self.n_occ} for {self.n_occ} occupied orbitals)")
        
    def build_fock(self, P: np.ndarray) -> tuple:
        """
        Build Fock matrix from density matrix.
        
        F = H_core + G[P]
        G = J - (1/2)K
        
        Args:
            P: Density matrix
        
        Returns:
            (F, J, K) tuple
        """
        n = self.basis.n_basis
        
        J = np.zeros((n, n))
        K = np.zeros((n, n))
        
        for mu in range(n):
            for nu in range(n):
                for lam in range(n):
                    for sig in range(n):
                        J[mu, nu] += P[lam, sig] * self.ERI[mu, nu, lam, sig]
                        K[mu, nu] += P[lam, sig] * self.ERI[mu, lam, nu, sig]
        
        G = J - 0.5 * K
        F = self.H_core + G
        
        return F, J, K
    
    def compute_energy(self, P: np.ndarray, F: np.ndarray) -> float:
        """
        Compute electronic energy.
        
        E_elec = (1/2) * Tr[P * (H_core + F)]
        
        Args:
            P: Density matrix
            F: Fock matrix
        
        Returns:
            Electronic energy
        """
        E_elec = 0.5 * np.sum(P * (self.H_core + F))
        return E_elec
    
    def compute_energy_components(self, P: np.ndarray, J: np.ndarray, K: np.ndarray) -> Dict[str, float]:
        """
        Compute energy decomposition.
        
        Returns:
            Dictionary with energy components
        """
        E_kinetic = np.sum(P * self.T)
        E_nuclear_attr = np.sum(P * self.V)
        E_coulomb = 0.5 * np.sum(P * J)
        E_exchange = -0.25 * np.sum(P * K)
        
        E_one_electron = E_kinetic + E_nuclear_attr
        E_two_electron = E_coulomb + E_exchange
        E_elec = E_one_electron + E_two_electron
        E_total = E_elec + self.E_nuc
        
        return {
            'E_kinetic': E_kinetic,
            'E_nuclear_attraction': E_nuclear_attr,
            'E_one_electron': E_one_electron,
            'E_coulomb': E_coulomb,
            'E_exchange': E_exchange,
            'E_two_electron': E_two_electron,
            'E_electronic': E_elec,
            'E_nuclear_repulsion': self.E_nuc,
            'E_total': E_total
        }
    
    def check_convergence(self, P_old: np.ndarray, P_new: np.ndarray,
                         F: np.ndarray, threshold_E: float = 1e-8,
                         threshold_D: float = 1e-6) -> tuple:
        """
        Check SCF convergence.
        
        Criteria:
        1. Energy change: |ΔE| < threshold_E
        2. Density change: RMS(ΔP) < threshold_D
        3. Commutator: ||FPS - SPF|| < threshold_D
        
        Args:
            P_old: Previous density matrix
            P_new: New density matrix
            F: Fock matrix
            threshold_E: Energy convergence threshold
            threshold_D: Density/commutator threshold
        
        Returns:
            (converged, delta_E, rms_D, max_commutator)
        """
        if len(self.energy_history) < 2:
            return False, 0.0, 0.0, 0.0
        
        delta_E = abs(self.energy_history[-1] - self.energy_history[-2])
        
        delta_P = P_new - P_old
        rms_D = np.sqrt(np.mean(delta_P**2))
        
        FPS = F @ P_new @ self.S
        SPF = self.S @ P_new @ F
        commutator = FPS - SPF
        max_commutator = np.max(np.abs(commutator))
        
        converged = (delta_E < threshold_E and 
                    rms_D < threshold_D and 
                    max_commutator < threshold_D)
        
        return converged, delta_E, rms_D, max_commutator
    
    def scf(self, max_iter: int = 100, 
            damping: float = 0.0,
            use_diis: bool = True,
            diis_start: int = 2,
            diis_size: int = 8,
            threshold_E: float = 1e-8,
            threshold_D: float = 1e-6,
            verbose: bool = True):
        """
        Run SCF iterations.
        
        Args:
            max_iter: Maximum number of iterations
            damping: Damping factor (0 = no damping, 0.2 = 20% old density)
            use_diis: Use DIIS acceleration
            diis_start: Iteration to start DIIS
            diis_size: DIIS subspace size
            threshold_E: Energy convergence threshold
            threshold_D: Density convergence threshold
            verbose: Print iteration details
        """
        if self.P is None:
            raise ValueError("Must call initial_guess() before scf()")
        
        print(f"\n{'='*60}")
        print("SCF Iterations")
        print(f"{'='*60}")
        print(f"Max iterations: {max_iter}")
        print(f"Damping factor: {damping}")
        print(f"DIIS: {'Yes' if use_diis else 'No'}")
        if use_diis:
            print(f"  Start at iteration: {diis_start}")
            print(f"  Subspace size: {diis_size}")
        print(f"Energy threshold: {threshold_E}")
        print(f"Density threshold: {threshold_D}")
        
        if use_diis:
            diis = DIIS(diis_size)
        
        print(f"\n{'Iter':>4} {'E(elec)':>16} {'E(total)':>16} {'ΔE':>12} {'RMS(ΔP)':>12} {'Max|[F,P]|':>12} {'Time':>8}")
        print("-" * 100)
        
        start_time = time.time()
        
        for iteration in range(1, max_iter + 1):
            iter_start = time.time()
            self.iteration = iteration
            
            P_old = self.P.copy()
            
            F, J, K = self.build_fock(self.P)
            
            if use_diis and iteration >= diis_start:
                error = F @ self.P @ self.S - self.S @ self.P @ F
                F = diis.update(F, error)
            
            self.eps, self.C = solve_generalized_eigenproblem(F, self.S, self.X)
            
            C_occ = self.C[:, :self.n_occ]
            P_new = 2.0 * C_occ @ C_occ.T
            
            if damping > 0:
                P_new = (1 - damping) * P_new + damping * P_old
            
            self.P = P_new
            self.F = F
            
            self.E_elec = self.compute_energy(self.P, F)
            self.E_total = self.E_elec + self.E_nuc
            self.energy_history.append(self.E_total)
            
            converged, delta_E, rms_D, max_comm = self.check_convergence(
                P_old, P_new, F, threshold_E, threshold_D)
            
            self.convergence_history.append({
                'iteration': iteration,
                'E_total': self.E_total,
                'delta_E': delta_E,
                'rms_D': rms_D,
                'max_commutator': max_comm
            })
            
            iter_time = time.time() - iter_start
            
            if verbose:
                print(f"{iteration:4d} {self.E_elec:16.10f} {self.E_total:16.10f} "
                      f"{delta_E:12.2e} {rms_D:12.2e} {max_comm:12.2e} {iter_time:8.2f}s")
            
            if converged:
                self.converged = True
                print(f"\n{'='*60}")
                print(f"SCF CONVERGED in {iteration} iterations!")
                print(f"{'='*60}")
                break
        
        else:
            print(f"\n{'='*60}")
            print(f"WARNING: SCF did NOT converge in {max_iter} iterations")
            print(f"{'='*60}")
        
        total_time = time.time() - start_time
        print(f"\nTotal SCF time: {total_time:.2f}s")
        
        self._print_results()
    
    def _print_results(self):
        """Print final results."""
        print(f"\n{'='*60}")
        print("Final Results")
        print(f"{'='*60}")
        
        components = self.compute_energy_components(self.P, *self.build_fock(self.P)[1:])
        
        print("\nEnergy Decomposition:")
        print(f"  Kinetic energy:           {components['E_kinetic']:16.10f} Eh")
        print(f"  Nuclear attraction:       {components['E_nuclear_attraction']:16.10f} Eh")
        print(f"  One-electron energy:      {components['E_one_electron']:16.10f} Eh")
        print(f"  Coulomb energy (J):       {components['E_coulomb']:16.10f} Eh")
        print(f"  Exchange energy (K):      {components['E_exchange']:16.10f} Eh")
        print(f"  Two-electron energy:      {components['E_two_electron']:16.10f} Eh")
        print(f"  Electronic energy:        {components['E_electronic']:16.10f} Eh")
        print(f"  Nuclear repulsion:        {components['E_nuclear_repulsion']:16.10f} Eh")
        print(f"  {'─'*50}")
        print(f"  Total energy:             {components['E_total']:16.10f} Eh")
        
        print("\nMolecular Orbital Energies:")
        print(f"  {'MO':>4} {'Occupancy':>10} {'Energy (Eh)':>14} {'Energy (eV)':>14}")
        print(f"  {'-'*50}")
        
        for i in range(len(self.eps)):
            occ = 2.0 if i < self.n_occ else 0.0
            label = ""
            if i == self.n_occ - 1:
                label = " (HOMO)"
            elif i == self.n_occ:
                label = " (LUMO)"
            
            print(f"  {i+1:4d} {occ:10.1f} {self.eps[i]:14.6f} {self.eps[i]*27.2114:14.6f}{label}")
        
        if self.n_occ < len(self.eps):
            homo_lumo_gap = (self.eps[self.n_occ] - self.eps[self.n_occ - 1]) * 27.2114
            print(f"\n  HOMO-LUMO gap: {homo_lumo_gap:.6f} eV")
    
    def get_results(self) -> Dict:
        """
        Get calculation results as dictionary.
        
        Returns:
            Dictionary with all results
        """
        if not self.converged:
            print("Warning: SCF not converged")
        
        components = self.compute_energy_components(self.P, *self.build_fock(self.P)[1:])
        
        return {
            'converged': self.converged,
            'iterations': self.iteration,
            'energy': self.E_total,
            'energy_components': components,
            'orbital_energies': self.eps,
            'mo_coefficients': self.C,
            'density_matrix': self.P,
            'fock_matrix': self.F,
            'overlap_matrix': self.S,
            'convergence_history': self.convergence_history
        }
