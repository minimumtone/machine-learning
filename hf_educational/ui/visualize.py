"""
Visualization functions for educational HF program.

Provides plots for:
- SCF convergence curves
- MO energy level diagrams
- Matrix heatmaps (S, H, J, K, F)
- Density slices
- Interactive J/K contribution sliders
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import List, Dict, Optional
import seaborn as sns


def plot_convergence(convergence_history: List[Dict], save_path: Optional[str] = None):
    """
    Plot SCF convergence curves.
    
    Shows energy and convergence metrics vs iteration.
    
    Args:
        convergence_history: List of convergence data from SCF
        save_path: Optional path to save figure
    """
    iterations = [d['iteration'] for d in convergence_history]
    energies = [d['E_total'] for d in convergence_history]
    delta_E = [d['delta_E'] for d in convergence_history]
    rms_D = [d['rms_D'] for d in convergence_history]
    max_comm = [d['max_commutator'] for d in convergence_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(iterations, energies, 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Total Energy (Eh)', fontsize=12)
    axes[0, 0].set_title('SCF Energy Convergence', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].semilogy(iterations[1:], delta_E[1:], 'o-', linewidth=2, markersize=6, color='orange')
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('|ΔE| (Eh)', fontsize=12)
    axes[0, 1].set_title('Energy Change', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=1e-8, color='r', linestyle='--', label='Threshold')
    axes[0, 1].legend()
    
    axes[1, 0].semilogy(iterations, rms_D, 'o-', linewidth=2, markersize=6, color='green')
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('RMS(ΔP)', fontsize=12)
    axes[1, 0].set_title('Density Matrix Change', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=1e-6, color='r', linestyle='--', label='Threshold')
    axes[1, 0].legend()
    
    axes[1, 1].semilogy(iterations, max_comm, 'o-', linewidth=2, markersize=6, color='purple')
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Max|[F,P]|', fontsize=12)
    axes[1, 1].set_title('Commutator Residual', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=1e-6, color='r', linestyle='--', label='Threshold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
    
    plt.show()


def plot_mo_diagram(eps: np.ndarray, n_occ: int, title: str = "Molecular Orbital Energies",
                   save_path: Optional[str] = None):
    """
    Plot MO energy level diagram.
    
    Shows occupied and unoccupied orbital energies with HOMO/LUMO gap.
    
    Args:
        eps: Orbital energies
        n_occ: Number of occupied orbitals
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    eps_eV = eps * 27.2114
    
    for i, energy in enumerate(eps_eV):
        if i < n_occ:
            color = 'blue'
            label = 'Occupied' if i == 0 else ''
        else:
            color = 'red'
            label = 'Unoccupied' if i == n_occ else ''
        
        ax.hlines(energy, i - 0.3, i + 0.3, colors=color, linewidth=3, label=label)
        
        if i == n_occ - 1:
            ax.text(i + 0.5, energy, 'HOMO', fontsize=10, fontweight='bold')
        elif i == n_occ:
            ax.text(i + 0.5, energy, 'LUMO', fontsize=10, fontweight='bold')
    
    if n_occ < len(eps):
        homo_lumo_gap = eps_eV[n_occ] - eps_eV[n_occ - 1]
        mid_energy = 0.5 * (eps_eV[n_occ] + eps_eV[n_occ - 1])
        ax.annotate('', xy=(n_occ - 0.5, eps_eV[n_occ]), 
                   xytext=(n_occ - 0.5, eps_eV[n_occ - 1]),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax.text(n_occ - 1.5, mid_energy, f'Gap\n{homo_lumo_gap:.2f} eV',
               fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    ax.set_xlabel('Molecular Orbital Index', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-1, len(eps))
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MO diagram saved to {save_path}")
    
    plt.show()


def plot_matrix_heatmap(matrix: np.ndarray, title: str, 
                       basis_labels: Optional[List[str]] = None,
                       save_path: Optional[str] = None):
    """
    Plot matrix as heatmap.
    
    Useful for visualizing S, H, J, K, F matrices.
    
    Args:
        matrix: Matrix to plot
        title: Plot title
        basis_labels: Optional labels for basis functions
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Basis Function', fontsize=12)
    ax.set_ylabel('Basis Function', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', fontsize=12)
    
    if basis_labels and len(basis_labels) <= 20:
        ax.set_xticks(range(len(basis_labels)))
        ax.set_yticks(range(len(basis_labels)))
        ax.set_xticklabels(basis_labels, rotation=45, ha='right')
        ax.set_yticklabels(basis_labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrix heatmap saved to {save_path}")
    
    plt.show()


def plot_density_slice(P: np.ndarray, basis, molecule, 
                      plane: str = 'xy', z_value: float = 0.0,
                      grid_points: int = 50, save_path: Optional[str] = None):
    """
    Plot 2D slice of electron density.
    
    Args:
        P: Density matrix
        basis: BasisSet object
        molecule: Molecule object
        plane: 'xy', 'xz', or 'yz'
        z_value: Value of the third coordinate
        grid_points: Number of grid points per dimension
        save_path: Optional path to save figure
    """
    coords = molecule.coords
    
    if plane == 'xy':
        x_min, x_max = coords[:, 0].min() - 3, coords[:, 0].max() + 3
        y_min, y_max = coords[:, 1].min() - 3, coords[:, 1].max() + 3
        x_grid = np.linspace(x_min, x_max, grid_points)
        y_grid = np.linspace(y_min, y_max, grid_points)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.full_like(X, z_value)
        xlabel, ylabel = 'X (Bohr)', 'Y (Bohr)'
        atom_x, atom_y = coords[:, 0], coords[:, 1]
    elif plane == 'xz':
        x_min, x_max = coords[:, 0].min() - 3, coords[:, 0].max() + 3
        z_min, z_max = coords[:, 2].min() - 3, coords[:, 2].max() + 3
        x_grid = np.linspace(x_min, x_max, grid_points)
        z_grid = np.linspace(z_min, z_max, grid_points)
        X, Z = np.meshgrid(x_grid, z_grid)
        Y = np.full_like(X, z_value)
        xlabel, ylabel = 'X (Bohr)', 'Z (Bohr)'
        atom_x, atom_y = coords[:, 0], coords[:, 2]
    else:
        y_min, y_max = coords[:, 1].min() - 3, coords[:, 1].max() + 3
        z_min, z_max = coords[:, 2].min() - 3, coords[:, 2].max() + 3
        y_grid = np.linspace(y_min, y_max, grid_points)
        z_grid = np.linspace(z_min, z_max, grid_points)
        Y, Z = np.meshgrid(y_grid, z_grid)
        X = np.full_like(Y, z_value)
        xlabel, ylabel = 'Y (Bohr)', 'Z (Bohr)'
        atom_x, atom_y = coords[:, 1], coords[:, 2]
    
    density = np.zeros_like(X)
    
    for i in range(grid_points):
        for j in range(grid_points):
            r = np.array([X[i, j], Y[i, j], Z[i, j]])
            
            chi = np.zeros(basis.n_basis)
            for mu in range(basis.n_basis):
                cgto = basis[mu].cgto
                for prim in cgto.primitives:
                    R = r - prim.center
                    r2 = np.dot(R, R)
                    chi[mu] += prim.coeff * (R[0]**prim.l * R[1]**prim.m * R[2]**prim.n *
                                            np.exp(-prim.alpha * r2))
            
            density[i, j] = np.dot(chi, P @ chi)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    levels = np.linspace(0, density.max(), 20)
    contour = ax.contourf(X if plane != 'yz' else Y, 
                         Y if plane == 'xy' else Z,
                         density, levels=levels, cmap='viridis')
    
    ax.scatter(atom_x, atom_y, c='red', s=200, marker='o', 
              edgecolors='white', linewidths=2, label='Atoms', zorder=5)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Electron Density ({plane.upper()} plane)', fontsize=14, fontweight='bold')
    ax.legend()
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Density', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Density slice saved to {save_path}")
    
    plt.show()


def create_interactive_jk_slider(H_core: np.ndarray, J: np.ndarray, K: np.ndarray,
                                 S: np.ndarray, X: np.ndarray, n_occ: int):
    """
    Create interactive plot with J/K contribution sliders.
    
    Educational tool to see how J and K contributions affect Fock matrix
    and orbital energies.
    
    Args:
        H_core: Core Hamiltonian
        J: Coulomb matrix
        K: Exchange matrix
        S: Overlap matrix
        X: Orthogonalization matrix
        n_occ: Number of occupied orbitals
    """
    from scipy import linalg
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.25)
    
    def compute_energy_and_plot(lambda_J, lambda_K):
        F = H_core + lambda_J * J - lambda_K * K
        
        F_prime = X.T @ F @ X
        eps, C_prime = linalg.eigh(F_prime)
        
        ax1.clear()
        ax2.clear()
        
        im = ax1.imshow(F, cmap='RdBu_r', aspect='auto')
        ax1.set_title(f'Fock Matrix\nλ_J={lambda_J:.2f}, λ_K={lambda_K:.2f}', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Basis Function')
        ax1.set_ylabel('Basis Function')
        
        eps_eV = eps * 27.2114
        for i, energy in enumerate(eps_eV[:min(15, len(eps))]):
            color = 'blue' if i < n_occ else 'red'
            ax2.hlines(energy, i - 0.3, i + 0.3, colors=color, linewidth=3)
        
        ax2.set_xlabel('Molecular Orbital')
        ax2.set_ylabel('Energy (eV)')
        ax2.set_title('MO Energies', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        fig.canvas.draw_idle()
    
    ax_lambda_J = plt.axes([0.15, 0.1, 0.65, 0.03])
    ax_lambda_K = plt.axes([0.15, 0.05, 0.65, 0.03])
    
    slider_J = Slider(ax_lambda_J, 'λ_J (Coulomb)', 0.0, 2.0, valinit=1.0)
    slider_K = Slider(ax_lambda_K, 'λ_K (Exchange)', 0.0, 2.0, valinit=1.0)
    
    def update(val):
        compute_energy_and_plot(slider_J.val, slider_K.val)
    
    slider_J.on_changed(update)
    slider_K.on_changed(update)
    
    compute_energy_and_plot(1.0, 1.0)
    
    plt.show()


def plot_all_matrices(S: np.ndarray, H: np.ndarray, J: np.ndarray, K: np.ndarray, F: np.ndarray,
                     basis_labels: Optional[List[str]] = None, save_dir: Optional[str] = None):
    """
    Plot all important matrices in a grid.
    
    Args:
        S, H, J, K, F: Matrices to plot
        basis_labels: Optional labels for basis functions
        save_dir: Optional directory to save figures
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    matrices = [
        (S, 'Overlap Matrix (S)', axes[0, 0]),
        (H, 'Core Hamiltonian (H)', axes[0, 1]),
        (J, 'Coulomb Matrix (J)', axes[0, 2]),
        (K, 'Exchange Matrix (K)', axes[1, 0]),
        (F, 'Fock Matrix (F)', axes[1, 1]),
        (J - 0.5*K, 'G Matrix (J - K/2)', axes[1, 2])
    ]
    
    for matrix, title, ax in matrices:
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Basis Function')
        ax.set_ylabel('Basis Function')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'all_matrices.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"All matrices plot saved to {save_path}")
    
    plt.show()
