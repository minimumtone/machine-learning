"""
Example: O2 molecule UHF calculation (triplet ground state).

Demonstrates open-shell UHF calculation with spin contamination analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from molecule_io.molecule import Molecule
from scf.uhf import UHF
from props.mulliken import mulliken_population_analysis, print_mulliken_analysis
from ui.visualize import plot_convergence, plot_mo_diagram


def main():
    """Run O2 UHF calculation with spin analysis."""
    
    print("\n" + "="*70)
    print(" O2 Molecule UHF Calculation - Educational Example")
    print(" Triplet Ground State (³Σg⁻)")
    print("="*70)
    
    o2 = Molecule.from_xyz_string("""2
O2 molecule at equilibrium bond length
O 0.0 0.0 0.0
O 0.0 0.0 1.21
""", charge=0, multiplicity=3)
    
    uhf = UHF(o2, 'sto-3g')
    
    uhf.compute_integrals(schwarz_threshold=1e-10)
    
    uhf.initial_guess('core')
    
    uhf.scf(max_iter=50, damping=0.2, use_diis=True, 
            diis_start=2, diis_size=8,
            threshold_E=1e-8, threshold_D=1e-6,
            verbose=True)
    
    if not uhf.converged:
        print("\nWARNING: SCF did not converge!")
        return
    
    results = uhf.get_results()
    
    P_total = uhf.P_alpha + uhf.P_beta
    mulliken_results = mulliken_population_analysis(P_total, uhf.S, uhf.basis, o2)
    print_mulliken_analysis(mulliken_results, o2)
    
    print("\n" + "="*70)
    print(" Spin Density Analysis")
    print("="*70)
    
    P_spin = uhf.P_alpha - uhf.P_beta
    spin_populations = []
    for atom_idx in range(o2.n_atoms):
        spin_pop = 0.0
        for mu in range(uhf.basis.n_basis):
            if uhf.basis[mu].atom_idx == atom_idx:
                for nu in range(uhf.basis.n_basis):
                    spin_pop += P_spin[mu, nu] * uhf.S[nu, mu]
        spin_populations.append(spin_pop)
    
    print("\nSpin populations:")
    for i, spin_pop in enumerate(spin_populations):
        print(f"  O{i+1}: {spin_pop:8.4f}")
    
    print("\n" + "="*70)
    print(" Generating Visualizations")
    print("="*70)
    
    print("\n1. Convergence curves...")
    plot_convergence(results['convergence_history'])
    
    print("\n2. Alpha MO energy diagram...")
    plot_mo_diagram(results['orbital_energies_alpha'], o2.n_alpha, 
                   title="O2 Alpha Molecular Orbital Energies (STO-3G)")
    
    print("\n3. Beta MO energy diagram...")
    plot_mo_diagram(results['orbital_energies_beta'], o2.n_beta, 
                   title="O2 Beta Molecular Orbital Energies (STO-3G)")
    
    print("\n" + "="*70)
    print(" Calculation Complete!")
    print("="*70)
    print(f"\nFinal Energy: {results['energy']:.10f} Eh")
    print(f"Iterations: {results['iterations']}")
    print(f"Converged: {results['converged']}")
    
    spin_props = results['spin_properties']
    print("\nSpin Properties:")
    print(f"  S_z = {spin_props['S_z']:.4f}")
    print(f"  <S²> expected = {spin_props['S2_expected']:.4f}")
    print(f"  <S²> computed = {spin_props['S2_computed']:.4f}")
    print(f"  Spin contamination = {spin_props['spin_contamination']:.4f}")


if __name__ == "__main__":
    main()
