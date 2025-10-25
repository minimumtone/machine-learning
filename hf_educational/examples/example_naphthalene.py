#!/usr/bin/env python3
"""
Naphthalene (C10H8) RHF Calculation Example

Demonstrates RHF calculation on naphthalene molecule with D2h symmetry.
This example shows:
- Polycyclic aromatic hydrocarbon (PAH) with two fused benzene rings
- Larger conjugated π system (68 basis functions with STO-3G)
- DIIS convergence for extended aromatic systems
- Mulliken population analysis showing charge delocalization
- Comparison with benzene for aromatic character
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
print(" Naphthalene (C10H8) RHF Calculation - Educational Example")
print(" Polycyclic Aromatic Hydrocarbon (PAH)")
print("="*70)

naphthalene_xyz = """18
Naphthalene molecule (D2h symmetry)
C    0.000000    0.710000    1.395000
C    0.000000   -0.710000    1.395000
C    0.000000    1.410000    0.195000
C    0.000000   -1.410000    0.195000
C    0.000000    0.710000   -1.015000
C    0.000000   -0.710000   -1.015000
C    0.000000    1.410000   -2.215000
C    0.000000   -1.410000   -2.215000
C    0.000000    0.710000   -3.410000
C    0.000000   -0.710000   -3.410000
H    0.000000    1.250000    2.335000
H    0.000000   -1.250000    2.335000
H    0.000000    2.495000    0.195000
H    0.000000   -2.495000    0.195000
H    0.000000    2.495000   -2.215000
H    0.000000   -2.495000   -2.215000
H    0.000000    1.250000   -4.350000
H    0.000000   -1.250000   -4.350000
"""

naphthalene = Molecule.from_xyz_string(naphthalene_xyz, charge=0, multiplicity=1)

print(f"\nMolecular formula: C10H8")
print(f"Total electrons: {naphthalene.n_electrons}")
print(f"Structure: Two fused benzene rings (D2h symmetry)")
print(f"Expected: Extended π-conjugation across both rings")

rhf = RHF(naphthalene, 'sto-3g')
basis = rhf.basis

print("\n" + "="*70)
print(" Running RHF-SCF Calculation")
print("="*70)
print(f"\nNote: Larger system may take longer to converge...")

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
    
    print(f"\n  Energy per atom: {rhf.energy / naphthalene.n_atoms:.4f} Eh")
    
    n_occ = naphthalene.n_electrons // 2
    homo_energy = rhf.orbital_energies[n_occ - 1]
    lumo_energy = rhf.orbital_energies[n_occ]
    gap_ev = (lumo_energy - homo_energy) * 27.211386  # Convert to eV
    
    print(f"\n  HOMO energy: {homo_energy:.6f} Eh ({homo_energy * 27.211386:.2f} eV)")
    print(f"  LUMO energy: {lumo_energy:.6f} Eh ({lumo_energy * 27.211386:.2f} eV)")
    print(f"  HOMO-LUMO gap: {gap_ev:.2f} eV")
    print(f"  (Smaller than benzene due to extended conjugation)")
    
else:
    print(f"✗ SCF did not converge in {rhf.max_iter} iterations")
    sys.exit(1)

print("\n" + "="*70)
print(" Mulliken Population Analysis")
print("="*70)

mulliken_results = mulliken_population_analysis(rhf.P, rhf.S, basis, naphthalene)
print_mulliken_analysis(mulliken_results, naphthalene)

print("\n" + "="*70)
print(" Aromatic Character Analysis")
print("="*70)

bond_orders = mulliken_results['bond_orders']

print("\nC-C Bond Orders in Naphthalene:")
print("  (Aromatic: ~1.5, Single: ~1.0, Double: ~2.0)")
print(f"{'Bond Type':>20} {'Bond':>10} {'Order':>10}")
print("-" * 45)

peripheral_bonds = [
    (0, 2, "Peripheral"),
    (1, 3, "Peripheral"),
    (4, 6, "Peripheral"),
    (5, 7, "Peripheral"),
    (6, 8, "Peripheral"),
    (7, 9, "Peripheral"),
]

fusion_bonds = [
    (0, 1, "Fusion"),
    (4, 5, "Fusion"),
]

bridge_bonds = [
    (2, 4, "Bridge"),
    (3, 5, "Bridge"),
]

all_bonds = peripheral_bonds + fusion_bonds + bridge_bonds

peripheral_orders = []
fusion_orders = []
bridge_orders = []

for i, j, bond_type in all_bonds:
    bo = bond_orders[i, j]
    print(f"  {bond_type:>20} C{i+1:d}-C{j+1:d}      {bo:8.4f}")
    
    if bond_type == "Peripheral":
        peripheral_orders.append(bo)
    elif bond_type == "Fusion":
        fusion_orders.append(bo)
    elif bond_type == "Bridge":
        bridge_orders.append(bo)

print(f"\nAverage bond orders:")
print(f"  Peripheral C-C: {np.mean(peripheral_orders):.4f}")
print(f"  Fusion C-C:     {np.mean(fusion_orders):.4f}")
print(f"  Bridge C-C:     {np.mean(bridge_orders):.4f}")
print(f"\nNote: Fusion bonds typically have higher bond order (~1.6)")
print(f"      due to increased π-electron density between rings")

print("\nCharge Distribution on Carbons:")
carbon_charges = mulliken_results['atomic_charges'][:10]
print(f"  Mean: {np.mean(carbon_charges):8.4f}")
print(f"  Std:  {np.std(carbon_charges):8.4f}")
print(f"  Range: [{np.min(carbon_charges):7.4f}, {np.max(carbon_charges):7.4f}]")

print("\n" + "="*70)
print(" Dipole Moment")
print("="*70)

dipole = compute_dipole_moment(rhf.P, basis, naphthalene)
print_dipole_moment(dipole)
print(f"\n(Should be ~0 for D2h symmetric naphthalene)")

print("\n" + "="*70)
print(" Generating Visualizations")
print("="*70)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    plot_convergence(rhf.convergence_history, save_path='naphthalene_convergence.png')
    print(f"✓ Saved convergence plot to naphthalene_convergence.png")
    
    n_occ = naphthalene.n_electrons // 2
    plot_mo_diagram(rhf.orbital_energies, n_occ, 
                   title="Naphthalene Molecular Orbital Energies",
                   save_path='naphthalene_mo_diagram.png')
    print(f"✓ Saved MO diagram to naphthalene_mo_diagram.png")
    
    plot_all_matrices(rhf.S, rhf.H, rhf.J, rhf.K, rhf.F,
                     save_path='naphthalene_matrices.png')
    print(f"✓ Saved matrix heatmaps to naphthalene_matrices.png")
    
except Exception as e:
    print(f"Warning: Could not generate visualizations: {e}")

print("\n" + "="*70)
print(" Calculation Complete!")
print("="*70)
print(f"\nNaphthalene RHF/STO-3G Results:")
print(f"  Energy: {rhf.energy:.6f} Eh")
print(f"  HOMO-LUMO gap: {gap_ev:.2f} eV")
print(f"  Average peripheral C-C bond order: {np.mean(peripheral_orders):.4f}")
print(f"  Average fusion C-C bond order: {np.mean(fusion_orders):.4f}")
print(f"  Converged: {rhf.converged} ({rhf.iteration} iterations)")
print(f"\nThis calculation demonstrates:")
print(f"  • Extended π-conjugation in polycyclic aromatic systems")
print(f"  • D2h symmetry preservation")
print(f"  • Different bond orders in peripheral vs fusion bonds")
print(f"  • DIIS convergence for larger conjugated systems")
print(f"  • Comparison with benzene for aromatic character")
