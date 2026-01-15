"""
SQD Band Gap Calculation Package

This package implements the Sample-based Quantum Diagonalization (SQD) method
for computing band gaps of periodic materials, based on the paper:
"Computing band gaps of periodic materials via sample-based quantum diagonalization"
(arXiv:2503.10901)

The package provides:
1. Classical preprocessing (DFT+U+V, tight-binding projection)
2. Extended Hubbard Hamiltonian construction
3. LUCJ ansatz circuit generation
4. Quantum circuit simulation (using ffsim)
5. SQD diagonalization and band gap calculation
"""

__version__ = "0.1.0"
