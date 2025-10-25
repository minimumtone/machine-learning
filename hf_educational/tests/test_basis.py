"""
Unit tests for basis set module.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from basis.basis_set import BasisSet, PrimitiveGTO, ContractedGTO
from basis.basis_parser import get_basis_data
from molecule_io.molecule import Molecule


def test_basis_normalization():
    """Test that basis functions are properly normalized."""
    print("\n" + "="*60)
    print("Test: Basis Function Normalization")
    print("="*60)
    
    h2 = Molecule.from_xyz_string("""2
H2 molecule
H 0.0 0.0 0.0
H 0.0 0.0 0.74
""", charge=0, multiplicity=1)
    
    basis = BasisSet(h2, 'sto-3g')
    
    from integrals.overlap import overlap_contracted
    
    print(f"\nBasis set: {basis.basis_name}")
    print(f"Number of basis functions: {basis.n_basis}")
    
    print("\nSelf-overlap integrals (should be ~1.0):")
    for i, bf in enumerate(basis.basis_functions):
        S_ii = overlap_contracted(bf.cgto, bf.cgto)
        print(f"  {bf.label}: {S_ii:.10f}")
        assert abs(S_ii - 1.0) < 1e-6, f"Basis function {bf.label} not normalized!"
    
    print("\n✓ All basis functions properly normalized")


def test_basis_symmetry():
    """Test that overlap matrix is symmetric."""
    print("\n" + "="*60)
    print("Test: Overlap Matrix Symmetry")
    print("="*60)
    
    h2o = Molecule.from_xyz_string("""3
Water molecule
O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0
""", charge=0, multiplicity=1)
    
    basis = BasisSet(h2o, 'sto-3g')
    
    from integrals.overlap import compute_overlap_matrix
    S = compute_overlap_matrix(basis)
    
    print(f"\nOverlap matrix shape: {S.shape}")
    print(f"Max asymmetry: {np.max(np.abs(S - S.T)):.2e}")
    
    assert np.allclose(S, S.T), "Overlap matrix not symmetric!"
    
    print("\n✓ Overlap matrix is symmetric")


def test_sto3g_data():
    """Test that STO-3G basis data is loaded correctly."""
    print("\n" + "="*60)
    print("Test: STO-3G Basis Data")
    print("="*60)
    
    for Z, symbol in [(1, 'H'), (6, 'C'), (8, 'O')]:
        data = get_basis_data(Z, 'sto-3g')
        print(f"\n{symbol} (Z={Z}):")
        for shell in data:
            print(f"  Shell: {shell['shell_type']}")
            print(f"  Exponents: {shell['exponents']}")
            print(f"  Coefficients: {shell['coefficients']}")
    
    print("\n✓ STO-3G basis data loaded successfully")


if __name__ == "__main__":
    test_basis_normalization()
    test_basis_symmetry()
    test_sto3g_data()
    print("\n" + "="*60)
    print("All basis tests passed!")
    print("="*60)
