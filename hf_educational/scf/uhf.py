"""
Unrestricted Hartree-Fock (UHF) implementation.

Implements open-shell UHF with:
- Separate alpha and beta spin orbitals
- SCF iterations with damping
- DIIS convergence acceleration
- Spin contamination analysis
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


class UHF:
    """
    Unrestricted Hartree-Fock for open-shell systems.
    
    Solves separate Roothaan-Hall equations for alpha and beta spins:
    F^α C^α = S C^α ε^α
    F^β C^β = S C^β ε^β
    
    where F^α = H_core + J[P^α + P^β] - K[P^α]
          F^β = H_core + J[P^α + P^β] - K[P^β]
    """
    
    def __init__(self, molecule: Molecule, basis_name: str = 'sto-3g'):
        """
        Initialize UHF calculation.
        
        Args:
            molecule: Molecule object
            basis_name: Basis set name
        """
        self.molecule = molecule
        self.basis_name = basis_name
        
        print(f"\n{'='*60}")
        print("Unrestricted Hartree-Fock (UHF) Calculation")
        print(f"{'='*60}")
        print(self.molecule)
        print(f"\nBasis set: {basis_name.upper()}")
        
        self.basis = BasisSet(molecule, basis_name)
        print(f"Number of basis functions: {self.basis.n_basis}")
        
        self.n_alpha = molecule.n_alpha
        self.n_beta = molecule.n_beta
        print(f"Number of alpha electrons: {self.n_alpha}")
        print(f"Number of beta electrons: {self.n_beta}")
        
        self.S = None
        self.T = None
        self.V = None
        self.H_core = None
        self.ERI = None
        self.X = None
        
        self.P_alpha = None
        self.P_beta = None
        self.F_alpha = None
        self.F_beta = None
        self.C_alpha = None
        self.C_beta = None
        self.eps_alpha = None
        self.eps_beta = None
        
        self.E_total = 0.0
        self.E_elec = 0.0
        self.E_nuc = 0.0
        
        self.converged = False
        self.iteration = 0
        
        self.energy_history = []
        self.convergence_history = []
        
    def compute_integrals(self, schwarz_threshold: float = 1e-10):
        """Compute all required integrals."""
        print(f"\n{'='*60}")
        print("Computing Integrals")
        print(f"{'='*60}")
        
        start = time.time()
        
        print("\n1. Overlap matrix S...")
        self.S = compute_overlap_matrix(self.basis)
        print(f"   Computed in {time.time() - start:.2f}s")
        
        start = time.time()
        print("\n2. Kinetic energy matrix T...")
        self.T = compute_kinetic_matrix(self.basis)
        print(f"   Computed in {time.time() - start:.2f}s")
        
        start = time.time()
        print("\n3. Nuclear attraction matrix V...")
        self.V = compute_nuclear_attraction_matrix(self.basis, self.molecule)
        print(f"   Computed in {time.time() - start:.2f}s")
        
        self.H_core = self.T + self.V
        
        self.E_nuc = self.molecule.nuclear_repulsion()
        print(f"\n4. Nuclear repulsion energy: {self.E_nuc:.10f} Eh")
        
        start = time.time()
        print("\n5. Two-electron integrals (ERI)...")
        self.ERI = compute_eri_tensor(self.basis, schwarz_threshold)
        print(f"   Computed in {time.time() - start:.2f}s")
        
        print("\n6. Symmetric orthogonalization X = S^(-1/2)...")
        self.X = symmetric_orthogonalization(self.S)
        
    def initial_guess(self, method: str = 'core'):
        """Generate initial density matrix guess."""
        print(f"\n{'='*60}")
        print(f"Initial Guess: {method}")
        print(f"{'='*60}")
        
        if method == 'core':
            eps, C = solve_generalized_eigenproblem(self.H_core, self.S, self.X)
            
            C_alpha_occ = C[:, :self.n_alpha]
            C_beta_occ = C[:, :self.n_beta]
            
            self.P_alpha = C_alpha_occ @ C_alpha_occ.T
            self.P_beta = C_beta_occ @ C_beta_occ.T
            
            print("Initial orbital energies (alpha occupied):")
            for i in range(self.n_alpha):
                print(f"  MO {i+1}: {eps[i]:12.6f} Eh")
            
            if self.n_beta < self.n_alpha:
                print("\nInitial orbital energies (beta occupied):")
                for i in range(self.n_beta):
                    print(f"  MO {i+1}: {eps[i]:12.6f} Eh")
        
        elif method == 'zero':
            self.P_alpha = np.zeros((self.basis.n_basis, self.basis.n_basis))
            self.P_beta = np.zeros((self.basis.n_basis, self.basis.n_basis))
        
        else:
            raise ValueError(f"Unknown initial guess method: {method}")
        
        print(f"\nAlpha density trace: {np.trace(self.P_alpha @ self.S):.6f} (should be {self.n_alpha})")
        print(f"Beta density trace: {np.trace(self.P_beta @ self.S):.6f} (should be {self.n_beta})")
        
    def build_fock(self, P_alpha: np.ndarray, P_beta: np.ndarray) -> tuple:
        """
        Build Fock matrices from density matrices.
        
        F^α = H_core + J[P^α + P^β] - K[P^α]
        F^β = H_core + J[P^α + P^β] - K[P^β]
        
        Args:
            P_alpha: Alpha density matrix
            P_beta: Beta density matrix
        
        Returns:
            (F_alpha, F_beta, J, K_alpha, K_beta) tuple
        """
        n = self.basis.n_basis
        P_total = P_alpha + P_beta
        
        J = np.zeros((n, n))
        K_alpha = np.zeros((n, n))
        K_beta = np.zeros((n, n))
        
        for mu in range(n):
            for nu in range(n):
                for lam in range(n):
                    for sig in range(n):
                        J[mu, nu] += P_total[lam, sig] * self.ERI[mu, nu, lam, sig]
                        K_alpha[mu, nu] += P_alpha[lam, sig] * self.ERI[mu, lam, nu, sig]
                        K_beta[mu, nu] += P_beta[lam, sig] * self.ERI[mu, lam, nu, sig]
        
        F_alpha = self.H_core + J - K_alpha
        F_beta = self.H_core + J - K_beta
        
        return F_alpha, F_beta, J, K_alpha, K_beta
    
    def compute_energy(self, P_alpha: np.ndarray, P_beta: np.ndarray,
                      F_alpha: np.ndarray, F_beta: np.ndarray) -> float:
        """
        Compute electronic energy.
        
        E_elec = (1/2) * [Tr(P^α * (H + F^α)) + Tr(P^β * (H + F^β))]
        
        Args:
            P_alpha, P_beta: Density matrices
            F_alpha, F_beta: Fock matrices
        
        Returns:
            Electronic energy
        """
        E_elec = 0.5 * (np.sum(P_alpha * (self.H_core + F_alpha)) +
                        np.sum(P_beta * (self.H_core + F_beta)))
        return E_elec
    
    def compute_spin_contamination(self) -> Dict[str, float]:
        """
        Compute spin contamination.
        
        <S^2> = S(S+1) for pure spin state
        <S^2> = S_z(S_z + 1) + N_β - sum_ij S_ij^αβ S_ji^βα
        
        Returns:
            Dictionary with spin properties
        """
        S_z = 0.5 * (self.n_alpha - self.n_beta)
        S_expected = S_z
        S2_expected = S_expected * (S_expected + 1)
        
        overlap_term = 0.0
        for i in range(self.n_alpha):
            for j in range(self.n_beta):
                S_ij = np.dot(self.C_alpha[:, i], self.S @ self.C_beta[:, j])
                overlap_term += S_ij ** 2
        
        S2_computed = S_z * (S_z + 1) + self.n_beta - overlap_term
        
        spin_contamination = S2_computed - S2_expected
        
        return {
            'S_z': S_z,
            'S_expected': S_expected,
            'S2_expected': S2_expected,
            'S2_computed': S2_computed,
            'spin_contamination': spin_contamination
        }
    
    def check_convergence(self, P_alpha_old: np.ndarray, P_alpha_new: np.ndarray,
                         P_beta_old: np.ndarray, P_beta_new: np.ndarray,
                         F_alpha: np.ndarray, F_beta: np.ndarray,
                         threshold_E: float = 1e-8,
                         threshold_D: float = 1e-6) -> tuple:
        """Check SCF convergence."""
        if len(self.energy_history) < 2:
            return False, 0.0, 0.0, 0.0
        
        delta_E = abs(self.energy_history[-1] - self.energy_history[-2])
        
        delta_P_alpha = P_alpha_new - P_alpha_old
        delta_P_beta = P_beta_new - P_beta_old
        rms_D = np.sqrt(0.5 * (np.mean(delta_P_alpha**2) + np.mean(delta_P_beta**2)))
        
        comm_alpha = F_alpha @ P_alpha_new @ self.S - self.S @ P_alpha_new @ F_alpha
        comm_beta = F_beta @ P_beta_new @ self.S - self.S @ P_beta_new @ F_beta
        max_comm = max(np.max(np.abs(comm_alpha)), np.max(np.abs(comm_beta)))
        
        converged = (delta_E < threshold_E and 
                    rms_D < threshold_D and 
                    max_comm < threshold_D)
        
        return converged, delta_E, rms_D, max_comm
    
    def scf(self, max_iter: int = 100, 
            damping: float = 0.0,
            use_diis: bool = True,
            diis_start: int = 2,
            diis_size: int = 8,
            threshold_E: float = 1e-8,
            threshold_D: float = 1e-6,
            verbose: bool = True):
        """Run SCF iterations."""
        if self.P_alpha is None or self.P_beta is None:
            raise ValueError("Must call initial_guess() before scf()")
        
        print(f"\n{'='*60}")
        print("SCF Iterations")
        print(f"{'='*60}")
        print(f"Max iterations: {max_iter}")
        print(f"Damping factor: {damping}")
        print(f"DIIS: {'Yes' if use_diis else 'No'}")
        
        if use_diis:
            diis_alpha = DIIS(diis_size)
            diis_beta = DIIS(diis_size)
        
        print(f"\n{'Iter':>4} {'E(elec)':>16} {'E(total)':>16} {'ΔE':>12} {'RMS(ΔP)':>12} {'Max|[F,P]|':>12} {'Time':>8}")
        print("-" * 100)
        
        start_time = time.time()
        
        for iteration in range(1, max_iter + 1):
            iter_start = time.time()
            self.iteration = iteration
            
            P_alpha_old = self.P_alpha.copy()
            P_beta_old = self.P_beta.copy()
            
            F_alpha, F_beta, J, K_alpha, K_beta = self.build_fock(self.P_alpha, self.P_beta)
            
            if use_diis and iteration >= diis_start:
                error_alpha = F_alpha @ self.P_alpha @ self.S - self.S @ self.P_alpha @ F_alpha
                error_beta = F_beta @ self.P_beta @ self.S - self.S @ self.P_beta @ F_beta
                F_alpha = diis_alpha.update(F_alpha, error_alpha)
                F_beta = diis_beta.update(F_beta, error_beta)
            
            self.eps_alpha, self.C_alpha = solve_generalized_eigenproblem(F_alpha, self.S, self.X)
            self.eps_beta, self.C_beta = solve_generalized_eigenproblem(F_beta, self.S, self.X)
            
            C_alpha_occ = self.C_alpha[:, :self.n_alpha]
            C_beta_occ = self.C_beta[:, :self.n_beta]
            
            P_alpha_new = C_alpha_occ @ C_alpha_occ.T
            P_beta_new = C_beta_occ @ C_beta_occ.T
            
            if damping > 0:
                P_alpha_new = (1 - damping) * P_alpha_new + damping * P_alpha_old
                P_beta_new = (1 - damping) * P_beta_new + damping * P_beta_old
            
            self.P_alpha = P_alpha_new
            self.P_beta = P_beta_new
            self.F_alpha = F_alpha
            self.F_beta = F_beta
            
            self.E_elec = self.compute_energy(self.P_alpha, self.P_beta, F_alpha, F_beta)
            self.E_total = self.E_elec + self.E_nuc
            self.energy_history.append(self.E_total)
            
            converged, delta_E, rms_D, max_comm = self.check_convergence(
                P_alpha_old, P_alpha_new, P_beta_old, P_beta_new,
                F_alpha, F_beta, threshold_E, threshold_D)
            
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
        
        print(f"\nTotal energy: {self.E_total:.10f} Eh")
        print(f"Electronic energy: {self.E_elec:.10f} Eh")
        print(f"Nuclear repulsion: {self.E_nuc:.10f} Eh")
        
        spin_props = self.compute_spin_contamination()
        print("\nSpin Properties:")
        print(f"  S_z = {spin_props['S_z']:.4f}")
        print(f"  <S^2> expected = {spin_props['S2_expected']:.4f}")
        print(f"  <S^2> computed = {spin_props['S2_computed']:.4f}")
        print(f"  Spin contamination = {spin_props['spin_contamination']:.4f}")
        
        print("\nAlpha Molecular Orbital Energies:")
        print(f"  {'MO':>4} {'Occupancy':>10} {'Energy (Eh)':>14} {'Energy (eV)':>14}")
        print(f"  {'-'*50}")
        
        for i in range(min(len(self.eps_alpha), 10)):
            occ = 1.0 if i < self.n_alpha else 0.0
            label = ""
            if i == self.n_alpha - 1:
                label = " (HOMO-α)"
            elif i == self.n_alpha:
                label = " (LUMO-α)"
            
            print(f"  {i+1:4d} {occ:10.1f} {self.eps_alpha[i]:14.6f} {self.eps_alpha[i]*27.2114:14.6f}{label}")
        
        print("\nBeta Molecular Orbital Energies:")
        print(f"  {'MO':>4} {'Occupancy':>10} {'Energy (Eh)':>14} {'Energy (eV)':>14}")
        print(f"  {'-'*50}")
        
        for i in range(min(len(self.eps_beta), 10)):
            occ = 1.0 if i < self.n_beta else 0.0
            label = ""
            if i == self.n_beta - 1:
                label = " (HOMO-β)"
            elif i == self.n_beta:
                label = " (LUMO-β)"
            
            print(f"  {i+1:4d} {occ:10.1f} {self.eps_beta[i]:14.6f} {self.eps_beta[i]*27.2114:14.6f}{label}")
    
    def get_results(self) -> Dict:
        """Get calculation results as dictionary."""
        if not self.converged:
            print("Warning: SCF not converged")
        
        spin_props = self.compute_spin_contamination()
        
        return {
            'converged': self.converged,
            'iterations': self.iteration,
            'energy': self.E_total,
            'orbital_energies_alpha': self.eps_alpha,
            'orbital_energies_beta': self.eps_beta,
            'mo_coefficients_alpha': self.C_alpha,
            'mo_coefficients_beta': self.C_beta,
            'density_matrix_alpha': self.P_alpha,
            'density_matrix_beta': self.P_beta,
            'fock_matrix_alpha': self.F_alpha,
            'fock_matrix_beta': self.F_beta,
            'overlap_matrix': self.S,
            'spin_properties': spin_props,
            'convergence_history': self.convergence_history
        }
