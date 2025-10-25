"""
Integration tests for RHF calculations.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from molecule_io.molecule import Molecule
from scf.rhf import RHF


def test_h2_rhf():
    """Test RHF on H2 molecule."""
    print("\n" + "="*60)
    print("Test: H2 RHF Calculation")
    print("="*60)
    
    h2 = Molecule.from_xyz_string("""2
H2 molecule at equilibrium
H 0.0 0.0 0.0
H 0.0 0.0 0.74
""", charge=0, multiplicity=1)
    
    rhf = RHF(h2, 'sto-3g')
    rhf.compute_integrals(schwarz_threshold=1e-10)
    rhf.initial_guess('core')
    rhf.scf(max_iter=50, damping=0.0, use_diis=True, verbose=True)
    
    assert rhf.converged, "H2 RHF did not converge"
    
    print(f"\nFinal energy: {rhf.E_total:.10f} Eh")
    
    expected_energy = -1.117  # Approximate STO-3G energy
    assert abs(rhf.E_total - expected_energy) < 0.01, f"H2 energy too far from expected"
    
    print("\n✓ H2 RHF test passed")
    return rhf


def test_he_rhf():
    """Test RHF on He atom (should converge in 1 iteration with core guess)."""
    print("\n" + "="*60)
    print("Test: He Atom RHF Calculation")
    print("="*60)
    
    he = Molecule.from_xyz_string("""1
Helium atom
He 0.0 0.0 0.0
""", charge=0, multiplicity=1)
    
    rhf = RHF(he, 'sto-3g')
    rhf.compute_integrals(schwarz_threshold=1e-10)
    rhf.initial_guess('core')
    rhf.scf(max_iter=50, damping=0.0, use_diis=False, verbose=True)
    
    assert rhf.converged, "He RHF did not converge"
    
    print(f"\nFinal energy: {rhf.E_total:.10f} Eh")
    print(f"Converged in {rhf.iteration} iterations")
    
    expected_energy = -2.8078  # STO-3G reference energy
    assert abs(rhf.E_total - expected_energy) < 0.001, f"He energy too far from expected"
    
    print("\n✓ He RHF test passed")
    return rhf


def test_h2o_rhf():
    """Test RHF on H2O molecule."""
    print("\n" + "="*60)
    print("Test: H2O RHF Calculation")
    print("="*60)
    
    h2o = Molecule.from_xyz_string("""3
Water molecule
O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0
""", charge=0, multiplicity=1)
    
    rhf = RHF(h2o, 'sto-3g')
    rhf.compute_integrals(schwarz_threshold=1e-10)
    rhf.initial_guess('core')
    rhf.scf(max_iter=50, damping=0.2, use_diis=True, verbose=True)
    
    assert rhf.converged, "H2O RHF did not converge"
    
    print(f"\nFinal energy: {rhf.E_total:.10f} Eh")
    
    expected_energy = -74.96  # Approximate STO-3G energy
    assert abs(rhf.E_total - expected_energy) < 0.1, f"H2O energy too far from expected"
    
    print("\n✓ H2O RHF test passed")
    return rhf


if __name__ == "__main__":
    test_h2_rhf()
    test_he_rhf()
    test_h2o_rhf()
    print("\n" + "="*60)
    print("All RHF tests passed!")
    print("="*60)
