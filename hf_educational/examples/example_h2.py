"""
Example: H2 molecule RHF calculation with STO-3G basis.

Demonstrates basic RHF calculation and visualization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from molecule_io.molecule import Molecule
from scf.rhf import RHF
from props.mulliken import mulliken_population_analysis, print_mulliken_analysis
from props.dipole import compute_dipole_moment, print_dipole_moment
from ui.visualize import plot_convergence, plot_mo_diagram, plot_all_matrices


def main():
    """Run H2 RHF calculation with full analysis."""
    
    print("\n" + "="*70)
    print(" H2 Molecule RHF Calculation - Educational Example")
    print("="*70)
    
    h2 = Molecule.from_xyz_string("""2
H2 molecule at equilibrium bond length
H 0.0 0.0 0.0
H 0.0 0.0 0.74
""", charge=0, multiplicity=1)
    
    rhf = RHF(h2, 'sto-3g')
    
    rhf.compute_integrals(schwarz_threshold=1e-10)
    
    rhf.initial_guess('core')
    
    rhf.scf(max_iter=50, damping=0.0, use_diis=True, 
            diis_start=2, diis_size=8,
            threshold_E=1e-8, threshold_D=1e-6,
            verbose=True)
    
    if not rhf.converged:
        print("\nWARNING: SCF did not converge!")
        return
    
    results = rhf.get_results()
    
    mulliken_results = mulliken_population_analysis(rhf.P, rhf.S, rhf.basis, h2)
    print_mulliken_analysis(mulliken_results, h2)
    
    dipole = compute_dipole_moment(rhf.P, rhf.basis, h2)
    print_dipole_moment(dipole)
    
    print("\n" + "="*70)
    print(" Generating Visualizations")
    print("="*70)
    
    plot_convergence(results['convergence_history'])
    
    plot_mo_diagram(results['orbital_energies'], h2.n_alpha, 
                   title="H2 Molecular Orbital Energies (STO-3G)")
    
    F, J, K = rhf.build_fock(rhf.P)
    basis_labels = [bf.label for bf in rhf.basis.basis_functions]
    plot_all_matrices(rhf.S, rhf.H_core, J, K, F, basis_labels=basis_labels)
    
    print("\n" + "="*70)
    print(" Calculation Complete!")
    print("="*70)
    print(f"\nFinal Energy: {results['energy']:.10f} Eh")
    print(f"Iterations: {results['iterations']}")
    print(f"Converged: {results['converged']}")


if __name__ == "__main__":
    main()
