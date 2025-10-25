"""
Example: H2O molecule RHF calculation with full visualization.

Demonstrates RHF calculation with convergence analysis, MO diagrams,
matrix visualization, and density plots.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from molecule_io.molecule import Molecule
from scf.rhf import RHF
from props.mulliken import mulliken_population_analysis, print_mulliken_analysis
from props.dipole import compute_dipole_moment, print_dipole_moment
from ui.visualize import (plot_convergence, plot_mo_diagram, plot_all_matrices,
                         plot_density_slice, create_interactive_jk_slider)


def main():
    """Run H2O RHF calculation with full analysis and visualization."""
    
    print("\n" + "="*70)
    print(" H2O Molecule RHF Calculation - Educational Example")
    print("="*70)
    
    h2o = Molecule.from_xyz_string("""3
Water molecule (C2v symmetry)
O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0
""", charge=0, multiplicity=1)
    
    rhf = RHF(h2o, 'sto-3g')
    
    rhf.compute_integrals(schwarz_threshold=1e-10)
    
    rhf.initial_guess('core')
    
    rhf.scf(max_iter=50, damping=0.2, use_diis=True, 
            diis_start=2, diis_size=8,
            threshold_E=1e-8, threshold_D=1e-6,
            verbose=True)
    
    if not rhf.converged:
        print("\nWARNING: SCF did not converge!")
        return
    
    results = rhf.get_results()
    
    mulliken_results = mulliken_population_analysis(rhf.P, rhf.S, rhf.basis, h2o)
    print_mulliken_analysis(mulliken_results, h2o)
    
    dipole = compute_dipole_moment(rhf.P, rhf.basis, h2o)
    print_dipole_moment(dipole)
    
    print("\n" + "="*70)
    print(" Generating Visualizations")
    print("="*70)
    
    print("\n1. Convergence curves...")
    plot_convergence(results['convergence_history'])
    
    print("\n2. MO energy diagram...")
    plot_mo_diagram(results['orbital_energies'], h2o.n_alpha, 
                   title="H2O Molecular Orbital Energies (STO-3G)")
    
    print("\n3. Matrix heatmaps...")
    F, J, K = rhf.build_fock(rhf.P)
    basis_labels = [bf.label for bf in rhf.basis.basis_functions]
    plot_all_matrices(rhf.S, rhf.H_core, J, K, F, basis_labels=basis_labels)
    
    print("\n4. Electron density slice (xy plane)...")
    plot_density_slice(rhf.P, rhf.basis, h2o, plane='xy', z_value=0.0, grid_points=50)
    
    print("\n5. Interactive J/K contribution slider...")
    print("   (Adjust sliders to see how Coulomb and Exchange affect MO energies)")
    create_interactive_jk_slider(rhf.H_core, J, K, rhf.S, rhf.X, h2o.n_alpha)
    
    print("\n" + "="*70)
    print(" Calculation Complete!")
    print("="*70)
    print(f"\nFinal Energy: {results['energy']:.10f} Eh")
    print(f"Iterations: {results['iterations']}")
    print(f"Converged: {results['converged']}")
    print(f"\nDipole moment: {dipole['mu_magnitude_debye']:.4f} Debye")


if __name__ == "__main__":
    main()
