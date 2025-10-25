"""
Unit tests for integral evaluation.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from basis.basis_set import BasisSet
from molecule_io.molecule import Molecule
from integrals import (compute_overlap_matrix, compute_kinetic_matrix,
                      compute_nuclear_attraction_matrix, compute_eri_tensor)
from integrals.boys import boys_function


def test_boys_function():
    """Test Boys function implementation."""
    print("\n" + "="*60)
    print("Test: Boys Function")
    print("="*60)
    
    test_cases = [
        (0, 0.0, 1.0),
        (0, 1.0, 0.7468241328124271),
        (1, 0.0, 1.0/3.0),
        (1, 1.0, 0.1894723458204924),
    ]
    
    print(f"\n{'n':>3} {'t':>8} {'F_n(t)':>15} {'Expected':>15} {'Error':>12}")
    print("-" * 60)
    
    for n, t, expected in test_cases:
        result = boys_function(n, t)
        error = abs(result - expected)
        print(f"{n:3d} {t:8.2f} {result:15.10f} {expected:15.10f} {error:12.2e}")
        assert error < 1e-8, f"Boys function error too large for F_{n}({t})"
    
    print("\n✓ Boys function tests passed")


def test_overlap_symmetry():
    """Test overlap integral symmetry."""
    print("\n" + "="*60)
    print("Test: Overlap Integral Symmetry")
    print("="*60)
    
    h2 = Molecule.from_xyz_string("""2
H2
H 0.0 0.0 0.0
H 0.0 0.0 0.74
""")
    
    basis = BasisSet(h2, 'sto-3g')
    S = compute_overlap_matrix(basis)
    
    print(f"\nOverlap matrix:\n{S}")
    print(f"\nMax asymmetry: {np.max(np.abs(S - S.T)):.2e}")
    
    assert np.allclose(S, S.T), "Overlap not symmetric"
    
    print("\n✓ Overlap symmetry test passed")


def test_kinetic_symmetry():
    """Test kinetic energy integral symmetry."""
    print("\n" + "="*60)
    print("Test: Kinetic Energy Integral Symmetry")
    print("="*60)
    
    h2 = Molecule.from_xyz_string("""2
H2
H 0.0 0.0 0.0
H 0.0 0.0 0.74
""")
    
    basis = BasisSet(h2, 'sto-3g')
    T = compute_kinetic_matrix(basis)
    
    print(f"\nKinetic matrix:\n{T}")
    print(f"\nMax asymmetry: {np.max(np.abs(T - T.T)):.2e}")
    
    assert np.allclose(T, T.T), "Kinetic not symmetric"
    assert np.all(np.diag(T) > 0), "Kinetic diagonal should be positive"
    
    print("\n✓ Kinetic symmetry test passed")


def test_nuclear_attraction():
    """Test nuclear attraction integrals."""
    print("\n" + "="*60)
    print("Test: Nuclear Attraction Integrals")
    print("="*60)
    
    h2 = Molecule.from_xyz_string("""2
H2
H 0.0 0.0 0.0
H 0.0 0.0 0.74
""")
    
    basis = BasisSet(h2, 'sto-3g')
    V = compute_nuclear_attraction_matrix(basis, h2)
    
    print(f"\nNuclear attraction matrix:\n{V}")
    print(f"\nMax asymmetry: {np.max(np.abs(V - V.T)):.2e}")
    
    assert np.allclose(V, V.T), "Nuclear attraction not symmetric"
    assert np.all(V < 0), "Nuclear attraction should be negative"
    
    print("\n✓ Nuclear attraction test passed")


def test_eri_symmetry():
    """Test ERI 8-fold symmetry."""
    print("\n" + "="*60)
    print("Test: ERI 8-fold Symmetry")
    print("="*60)
    
    he = Molecule.from_xyz_string("""1
He
He 0.0 0.0 0.0
""")
    
    basis = BasisSet(he, 'sto-3g')
    print(f"\nComputing ERI for He (STO-3G)...")
    print(f"Basis functions: {basis.n_basis}")
    
    ERI = compute_eri_tensor(basis, schwarz_threshold=1e-12)
    
    n = basis.n_basis
    max_error = 0.0
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    val = ERI[i, j, k, l]
                    
                    symmetries = [
                        ERI[j, i, k, l],
                        ERI[i, j, l, k],
                        ERI[j, i, l, k],
                        ERI[k, l, i, j],
                        ERI[l, k, i, j],
                        ERI[k, l, j, i],
                        ERI[l, k, j, i]
                    ]
                    
                    for sym_val in symmetries:
                        error = abs(val - sym_val)
                        max_error = max(max_error, error)
    
    print(f"\nMax symmetry violation: {max_error:.2e}")
    assert max_error < 1e-10, "ERI symmetry violated"
    
    print("\n✓ ERI symmetry test passed")


if __name__ == "__main__":
    test_boys_function()
    test_overlap_symmetry()
    test_kinetic_symmetry()
    test_nuclear_attraction()
    test_eri_symmetry()
    print("\n" + "="*60)
    print("All integral tests passed!")
    print("="*60)
