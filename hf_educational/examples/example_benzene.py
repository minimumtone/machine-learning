#!/usr/bin/env python3
"""
Benzene (C6H6) RHF Calculation Example

Demonstrates RHF calculation on benzene molecule with D6h symmetry.
This example shows:
- Aromatic system with delocalized π electrons
- Larger basis set (42 basis functions with STO-3G)
- DIIS convergence for conjugated systems
- Mulliken population analysis showing charge distribution
- Visualization of MO energies and aromatic character
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time

from molecule_io.molecule import Molecule
from basis.basis_set import BasisSet
from scf.rhf import RHF
from props.mulliken import mulliken_population_analysis, print_mulliken_analysis
from props.dipole import compute_dipole_moment, print_dipole_moment
from ui.visualize import plot_convergence, plot_mo_diagram, plot_all_matrices

print("\n" + "="*70)
print(" Benzene (C6H6) RHF Calculation - Educational Example")
print("="*70)

benzene_xyz = """12
Benzene molecule (D6h symmetry)
C    1.390000    0.000000    0.000000
C    0.695000    1.203700    0.000000
C   -0.695000    1.203700    0.000000
C   -1.390000    0.000000    0.000000
C   -0.695000   -1.203700    0.000000
C    0.695000   -1.203700    0.000000
H    2.471000    0.000000    0.000000
H    1.235500    2.139700    0.000000
H   -1.235500    2.139700    0.000000
H   -2.471000    0.000000    0.000000
H   -1.235500   -2.139700    0.000000
H    1.235500   -2.139700    0.000000
"""

benzene = Molecule.from_xyz_string(benzene_xyz, charge=0, multiplicity=1)

print("\nMolecular formula: C6H6")
print(f"Total electrons: {benzene.n_electrons}")
print("Expected aromatic character with delocalized π system")

rhf = RHF(benzene, 'sto-3g')
basis = rhf.basis

print("\n" + "="*70)
print(" Running RHF-SCF Calculation")
print("="*70)

start_time = time.time()
rhf.compute_integrals(schwarz_threshold=1e-10)
rhf.initial_guess('core')
rhf.scf(max_iter=100, damping=0.2, use_diis=True, 
        diis_start=2, diis_size=8,
        threshold_E=1e-8, threshold_D=1e-6,
        verbose=True)
calc_time = time.time() - start_time

print("\n" + "="*70)
print(" Results Summary")
print("="*70)

if rhf.converged:
    print(f"✓ SCF converged in {rhf.iteration} iterations")
    print(f"  Calculation time: {calc_time:.2f} seconds")
    print(f"\n  Final energy: {rhf.energy:.8f} Eh")
    print(f"  Electronic energy: {rhf.energy - rhf.E_nuc:.8f} Eh")
    print(f"  Nuclear repulsion: {rhf.E_nuc:.8f} Eh")
    
    print(f"\n  Energy per atom: {rhf.energy / benzene.n_atoms:.4f} Eh")
    
    n_occ = benzene.n_electrons // 2
    homo_energy = rhf.orbital_energies[n_occ - 1]
    lumo_energy = rhf.orbital_energies[n_occ]
    gap_ev = (lumo_energy - homo_energy) * 27.211386  # Convert to eV
    
    print(f"\n  HOMO energy: {homo_energy:.6f} Eh ({homo_energy * 27.211386:.2f} eV)")
    print(f"  LUMO energy: {lumo_energy:.6f} Eh ({lumo_energy * 27.211386:.2f} eV)")
    print(f"  HOMO-LUMO gap: {gap_ev:.2f} eV")
    print("  (Typical for aromatic systems: 5-8 eV)")
    
else:
    print(f"✗ SCF did not converge in {rhf.max_iter} iterations")
    sys.exit(1)

print("\n" + "="*70)
print(" Mulliken Population Analysis")
print("="*70)

mulliken_results = mulliken_population_analysis(rhf.P, rhf.S, basis, benzene)
print_mulliken_analysis(mulliken_results, benzene)

print("\n" + "="*70)
print(" Aromatic Character Analysis")
print("="*70)

bond_orders = mulliken_results['bond_orders']
print("\nC-C Bond Orders (aromatic character):")
print("  (Perfect aromatic: ~1.5, single: ~1.0, double: ~2.0)")
print(f"{'Bond':>12} {'Order':>10}")
print("-" * 25)

carbon_indices = [i for i in range(6)]  # First 6 atoms are carbons
for i in range(6):
    j = (i + 1) % 6  # Next carbon in ring
    bo = bond_orders[i, j]
    print(f"  C{i+1}-C{j+1:d}      {bo:8.4f}")

avg_cc_bond_order = np.mean([bond_orders[i, (i+1)%6] for i in range(6)])
print(f"\nAverage C-C bond order: {avg_cc_bond_order:.4f}")
print(f"(Ideal aromatic: 1.5, observed: {avg_cc_bond_order:.4f})")

print("\nCharge Distribution on Carbons:")
carbon_charges = mulliken_results['atomic_charges'][:6]
print(f"  Mean: {np.mean(carbon_charges):8.4f}")
print(f"  Std:  {np.std(carbon_charges):8.4f}")
print("  (Should be near zero and uniform for symmetric benzene)")

print("\n" + "="*70)
print(" Dipole Moment")
print("="*70)

dipole = compute_dipole_moment(rhf.P, basis, benzene)
print_dipole_moment(dipole)
print("\n(Should be ~0 for D6h symmetric benzene)")

print("\n" + "="*70)
print(" Generating Visualizations")
print("="*70)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    plot_convergence(rhf.convergence_history, save_path='benzene_convergence.png')
    print("✓ Saved convergence plot to benzene_convergence.png")
    
    n_occ = benzene.n_electrons // 2
    plot_mo_diagram(rhf.orbital_energies, n_occ, 
                   title="Benzene Molecular Orbital Energies",
                   save_path='benzene_mo_diagram.png')
    print("✓ Saved MO diagram to benzene_mo_diagram.png")
    
    plot_all_matrices(rhf.S, rhf.H, rhf.J, rhf.K, rhf.F,
                     save_path='benzene_matrices.png')
    print("✓ Saved matrix heatmaps to benzene_matrices.png")
    
except Exception as e:
    print(f"Warning: Could not generate visualizations: {e}")

print("\n" + "="*70)
print(" Calculation Complete!")
print("="*70)
print("\nBenzene RHF/STO-3G Results:")
print(f"  Energy: {rhf.energy:.6f} Eh")
print(f"  HOMO-LUMO gap: {gap_ev:.2f} eV")
print(f"  Average C-C bond order: {avg_cc_bond_order:.4f}")
print(f"  Converged: {rhf.converged} ({rhf.iteration} iterations)")
print("\nThis calculation demonstrates:")
print("  • Aromatic π-electron delocalization")
print("  • D6h symmetry preservation")
print("  • DIIS convergence for conjugated systems")
print("  • Mulliken analysis of aromatic character")
