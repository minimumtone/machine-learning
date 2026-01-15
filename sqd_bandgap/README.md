# SQD Band Gap Calculation

This package implements the Sample-based Quantum Diagonalization (SQD) method for computing band gaps of periodic materials.

## Overview

The SQD method is a quantum-classical hybrid approach that combines:
1. **Classical preprocessing**: DFT+U+V calculations and tight-binding projection
2. **Quantum sampling**: LUCJ ansatz circuit execution on quantum hardware/simulator
3. **Classical post-processing**: SQD diagonalization in the sampled configuration space

## Method

The band gap is calculated using the formula:

```
Eg = E[Ne-1] + E[Ne+1] - 2*E[Ne]
```

where:
- `E[Ne]` is the ground state energy with Ne electrons (neutral system)
- `E[Ne-1]` is the energy with one electron removed (cation)
- `E[Ne+1]` is the energy with one electron added (anion)

## Package Structure

```
sqd_bandgap/
├── __init__.py          # Package initialization
├── hamiltonian.py       # Extended Hubbard Hamiltonian construction
├── classical.py         # Classical preprocessing (HF, CCSD)
├── circuits.py          # Quantum circuit generation (LUCJ ansatz)
├── sqd.py               # SQD algorithm implementation
├── bandgap.py           # Band gap calculation utilities
├── run_bandgap_calculation.py  # Main execution script
└── README.md            # This file
```

## Requirements

- Python >= 3.9
- numpy
- pyscf
- ffsim
- qiskit
- qiskit-addon-sqd

## Installation

```bash
pip install numpy pyscf ffsim qiskit qiskit-addon-sqd
```

## Usage

### Quick Demo (using pre-computed results)

```bash
cd sqd_bandgap
python run_bandgap_calculation.py --material hafnium_2
```

### With Custom Results

```bash
python run_bandgap_calculation.py \
    --material hafnium_2 \
    --results_path /path/to/sqd-band-gaps/sqd-lattice/runs \
    --ne_neutral 24 \
    --sampling hardware
```

## Example Output

```
============================================================
SQD Band Gap Calculation for HfO2
============================================================
  23e: E = -389.537290 eV
  24e: E = -389.433096 eV
  25e: E = -383.666757 eV

============================================================
Band Gap Calculation
============================================================
Formula: Eg = E[Ne-1] + E[Ne+1] - 2*E[Ne]
       = -389.537290 + -383.666757 - 2*-389.433096
       = 5.6621 eV

============================================================
Comparison with Reference Values
============================================================
SQD Band Gap:          5.6621 eV
Experimental:          5.7000 eV
SQD Error:             0.0379 eV (0.7%)
DFT+U+V Band Gap:      4.5000 eV
DFT+U+V Error:         1.2000 eV (21.1%)
SQD Improvement:       1.1621 eV (96.8% reduction)
```

## Results

For HfO2 (hafnium dioxide):
- **SQD Band Gap**: 5.66 eV
- **Experimental**: 5.7 eV
- **DFT+U+V**: 4.5 eV

SQD achieves ~97% reduction in error compared to DFT+U+V.

## Reference

Based on the paper:
> "Computing band gaps of periodic materials via sample-based quantum diagonalization"
> arXiv:2503.10901

Original code repository: https://github.com/neumannrf/sqd-band-gaps

## License

MIT License
