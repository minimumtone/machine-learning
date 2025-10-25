ｎ# LCAO-Hartree-Fock Educational Program

A comprehensive implementation of Restricted and Unrestricted Hartree-Fock methods for educational purposes. This package provides a complete, theory-driven implementation built from scratch to help understand quantum chemistry calculations.

## Features

### Core Capabilities
- **Restricted Hartree-Fock (RHF)** for closed-shell systems
- **Unrestricted Hartree-Fock (UHF)** for open-shell systems
- Gaussian basis sets (STO-3G included, extensible to 6-31G, etc.)
- Full integral evaluation from scratch using Obara-Saika recursion
- DIIS convergence acceleration
- Schwarz screening for efficient ERI evaluation

### Analysis Tools
- Mulliken population analysis
- Dipole moment calculation
- Energy decomposition (kinetic, nuclear attraction, Coulomb, exchange)
- Spin contamination analysis (UHF)

### Educational Visualizations
- SCF convergence curves (energy, density change, commutator)
- Molecular orbital energy level diagrams (HOMO/LUMO gap)
- Matrix heatmaps (S, H, J, K, F)
- Electron density slices (2D)
- Interactive J/K contribution sliders

## Theoretical Background

### Hartree-Fock Theory

The Hartree-Fock method approximates the many-electron wavefunction as a single Slater determinant, leading to the self-consistent field (SCF) equations.

#### Roothaan-Hall Equations (RHF)

For closed-shell systems, the spatial orbitals φᵢ are expanded in atomic orbitals (LCAO):

```
φᵢ(r) = Σμ Cμᵢ χμ(r)
```

This leads to the matrix eigenvalue problem:

```
F C = S C ε
```

where:
- **F** is the Fock matrix: `F = H_core + G[P]`
- **H_core** is the core Hamiltonian: `H_core = T + V`
- **G** is the two-electron part: `G = J - (1/2)K`
- **S** is the overlap matrix
- **C** contains MO coefficients
- **ε** are orbital energies

#### Density Matrix

For closed-shell RHF:

```
Pμν = 2 Σᵢ₌₁ⁿᵒᶜᶜ Cμᵢ Cνᵢ
```

#### Coulomb and Exchange Matrices

```
Jμν = Σλσ Pλσ (μν|λσ)
Kμν = Σλσ Pλσ (μλ|νσ)
```

where `(μν|λσ)` are two-electron repulsion integrals (ERIs).

#### Total Energy

```
E_elec = Σμν Pμν (Hμν + (1/2)Gμν)
E_total = E_elec + E_nn
```

### Unrestricted Hartree-Fock (UHF)

For open-shell systems, separate spatial orbitals for α and β spins:

```
F^α = H_core + J[P^α + P^β] - K[P^α]
F^β = H_core + J[P^α + P^β] - K[P^β]
```

### Integral Evaluation

#### Gaussian-Type Orbitals (GTOs)

Primitive GTO:
```
g(r) = N x^l y^m z^n exp(-α|r-A|²)
```

Contracted GTO:
```
χ(r) = Σₚ dₚ gₚ(r)
```

#### Boys Function

The Boys function is central to nuclear attraction and ERI evaluation:

```
Fₙ(t) = ∫₀¹ u^(2n) exp(-t u²) du
```

Related to incomplete gamma function:
```
Fₙ(t) = (1/2) t^(-n-1/2) γ(n+1/2, t)
```

#### Obara-Saika Recursion

Integrals are evaluated using recursion relations that build up from s-type GTOs to higher angular momentum.

### DIIS Acceleration

Direct Inversion in the Iterative Subspace (DIIS) accelerates SCF convergence by extrapolating the Fock matrix using error vectors:

```
e = FPS - SPF  (commutator)
```

DIIS finds optimal linear combination of previous Fock matrices to minimize error.

## Installation

```bash
cd hf_educational
pip install -r requirements.txt
```

## Usage

### Basic RHF Calculation

```python
from io.molecule import Molecule
from scf.rhf import RHF

# Define molecule
h2o = Molecule.from_xyz_string("""3
Water molecule
O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0
""", charge=0, multiplicity=1)

# Initialize RHF
rhf = RHF(h2o, 'sto-3g')

# Compute integrals
rhf.compute_integrals()

# Initial guess
rhf.initial_guess('core')

# Run SCF
rhf.scf(max_iter=50, damping=0.2, use_diis=True)

# Get results
results = rhf.get_results()
print(f"Total energy: {results['energy']:.10f} Eh")
```

### UHF Calculation

```python
from scf.uhf import UHF

# O2 triplet ground state
o2 = Molecule.from_xyz_string("""2
O2 molecule
O 0.0 0.0 0.0
O 0.0 0.0 1.21
""", charge=0, multiplicity=3)

uhf = UHF(o2, 'sto-3g')
uhf.compute_integrals()
uhf.initial_guess('core')
uhf.scf(max_iter=50, damping=0.2, use_diis=True)

# Spin contamination analysis
spin_props = uhf.compute_spin_contamination()
print(f"<S²> = {spin_props['S2_computed']:.4f}")
```

### Population Analysis

```python
from props.mulliken import mulliken_population_analysis, print_mulliken_analysis

mulliken_results = mulliken_population_analysis(rhf.P, rhf.S, rhf.basis, molecule)
print_mulliken_analysis(mulliken_results, molecule)
```

### Visualization

```python
from ui.visualize import plot_convergence, plot_mo_diagram, plot_all_matrices

# Convergence curves
plot_convergence(results['convergence_history'])

# MO diagram
plot_mo_diagram(results['orbital_energies'], molecule.n_alpha)

# Matrix heatmaps
F, J, K = rhf.build_fock(rhf.P)
plot_all_matrices(rhf.S, rhf.H_core, J, K, F)
```

## Examples

Run the provided examples:

```bash
# H2 molecule
python examples/example_h2.py

# H2O molecule with full visualization
python examples/example_h2o.py

# O2 molecule (UHF)
python examples/example_o2_uhf.py
```

## Testing

Run unit tests:

```bash
# Test basis sets
python tests/test_basis.py

# Test integrals
python tests/test_integrals.py

# Test RHF
python tests/test_rhf.py
```

## Project Structure

```
hf_educational/
├── __init__.py
├── basis/              # Basis set definitions and parsing
│   ├── basis_set.py
│   └── basis_parser.py
├── integrals/          # Integral evaluation
│   ├── overlap.py
│   ├── kinetic.py
│   ├── nuclear.py
│   ├── eri.py
│   └── boys.py
├── linalg/             # Linear algebra utilities
│   ├── orthogonalize.py
│   └── diagonalize.py
├── scf/                # SCF methods
│   ├── rhf.py
│   ├── uhf.py
│   └── diis.py
├── props/              # Properties and analysis
│   ├── mulliken.py
│   └── dipole.py
├── io/                 # Input/output
│   └── molecule.py
├── ui/                 # Visualization
│   └── visualize.py
├── tests/              # Unit tests
│   ├── test_basis.py
│   ├── test_integrals.py
│   └── test_rhf.py
└── examples/           # Example calculations
    ├── example_h2.py
    ├── example_h2o.py
    └── example_o2_uhf.py
```

## Educational Features

### Energy Decomposition

The program provides detailed energy breakdown:
- Kinetic energy (T)
- Nuclear attraction (V)
- One-electron energy (T + V)
- Coulomb energy (J)
- Exchange energy (K)
- Two-electron energy (J - K/2)

### Interactive J/K Sliders

Adjust Coulomb (J) and Exchange (K) contributions interactively to see their effect on:
- Fock matrix structure
- Orbital energies
- HOMO-LUMO gap

This helps understand the physical meaning of these terms.

### Convergence Monitoring

Track SCF convergence with multiple metrics:
- Total energy change (ΔE)
- Density matrix change (RMS ΔP)
- Commutator residual (||[F,P]||)

## Theory References

1. **Szabo, A.; Ostlund, N. S.** *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*. Dover, 1996.

2. **Helgaker, T.; Jørgensen, P.; Olsen, J.** *Molecular Electronic-Structure Theory*. Wiley, 2000.

3. **Pulay, P.** Convergence acceleration of iterative sequences. The case of SCF iteration. *Chem. Phys. Lett.* **1980**, *73*, 393-398.

4. **Obara, S.; Saika, A.** Efficient recursive computation of molecular integrals over Cartesian Gaussian functions. *J. Chem. Phys.* **1986**, *84*, 3963-3974.

## Limitations

- Currently supports only Cartesian GTOs (not spherical harmonics)
- STO-3G basis set included; other basis sets need to be added
- No geometry optimization (fixed geometries only)
- No post-HF methods (MP2, CI, CC)
- ERI evaluation can be slow for large basis sets (educational implementation)

## Future Extensions

- Additional basis sets (6-31G, 6-31G*, cc-pVDZ, etc.)
- Spherical harmonic GTOs
- Geometry optimization
- Analytical gradients
- Vibrational frequencies
- Post-HF methods (MP2)
- Density functional theory (DFT)

## License

Educational use. See LICENSE file for details.

## Acknowledgments

This implementation follows standard quantum chemistry textbooks and is designed for educational purposes to help students understand the theory and implementation of Hartree-Fock methods.
