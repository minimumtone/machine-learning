# δ-Ni₂Si Antisite Defect Formation Energy Analysis

## Overview

This module implements a comprehensive workflow for calculating antisite defect formation energies in δ-Ni₂Si (Pnma structure) with Short-Range Order (SRO) analysis using KL divergence.

## Crystal Structure

### Space Group and Lattice Parameters

δ-Ni₂Si crystallizes in the orthorhombic Pnma space group (#62) with:

- **Lattice constants**: a = 5.00 Å, b = 3.73 Å, c = 7.04 Å
- **2×2×1 Supercell**: 10.00 × 7.46 × 7.04 Å
- **Total atoms**: 48 (stoichiometric: 32 Ni + 16 Si)

### Wyckoff 4c Positions

| Site | x | y | z | Ideal Species |
|------|-------|-------|-------|---------------|
| Ni1 | 0.038 | 0.250 | 0.218 | Ni |
| Ni2 | 0.183 | 0.250 | 0.561 | Ni |
| Si1 | 0.712 | 0.250 | 0.611 | Si |

Each Wyckoff 4c position generates 4 equivalent sites through space group operations:
1. (x, 1/4, z)
2. (1/2-x, 3/4, 1/2+z)
3. (-x, 3/4, -z)
4. (1/2+x, 1/4, 1/2-z)

## Structure Sampling (Large-Scale Selection Workflow)

The workflow generates a large candidate pool and selects the top 256 structures with highest short-range order (KL divergence).

### Candidate Generation Strategies

For each composition (Ni₃₂Si₁₆, Ni₃₄Si₁₄, Ni₃₀Si₁₈), candidates are generated using multiple strategies:

| Strategy | Proportion | Description |
|----------|------------|-------------|
| Random | 40% | Uniform random configurations (also used for baseline Q(σ)) |
| SA (pair potential) | 30% | Simulated annealing with pair potential minimization |
| Site-specific | 20% | Targeted substitutions on specific sublattices |
| Random fill | 10% | Additional random to ensure diversity |

### Pair Potential Hamiltonian

```
H = Σ_{i<j} ε_αβ / r_ij^k
```

Parameters:
- ε_Ni-Si = -1.0 (attractive)
- ε_Ni-Ni = 0.5 (repulsive)
- ε_Si-Si = 0.8 (repulsive)
- Cutoff: r_ij < 3.5 Å
- Default k = 1 (configurable)

### Selection Process

1. Generate large candidate pool (default: 5000 per composition = 15000 total)
2. Calculate composition-specific baseline Q(σ) from random configurations
3. Calculate KL divergence for all candidates against their composition's baseline
4. Rank all candidates by KL divergence (descending)
5. Select top 256 structures with highest ordering

### Compositions

| Composition | Description |
|-------------|-------------|
| Ni₃₂Si₁₆ | Stoichiometric |
| Ni₃₄Si₁₄ | Ni-rich (2 excess Ni) |
| Ni₃₀Si₁₈ | Si-rich (2 excess Si) |

### Legacy Workflow (Group-based)

The original group-based workflow is still available via `run_full_workflow()`:
- Group A: 100 random baseline structures
- Group B: 100 SA-optimized structures  
- Group C: 56 specific defect configurations

## VASP Calculation Parameters

### INCAR Settings

```ini
PREC   = Accurate
ENCUT  = 520
EDIFF  = 1E-6
EDIFFG = -0.02
ISMEAR = 0
SIGMA  = 0.05
LREAL  = Auto
ISPIN  = 2
MAGMOM = 32*0.6 16*0.0
LWAVE  = .TRUE.
LCHARG = .TRUE.
NCORE  = 4
```

### KPOINTS

Gamma-centered 4×4×4 mesh (adjust based on supercell size)

### Important Notes

1. **MAGMOM ordering**: POSCAR files are generated with Ni atoms first, then Si atoms, matching the MAGMOM specification
2. **Magnetic convergence**: If Ni moments collapse (< 0.05 μB), flag for non-magnetic analysis
3. **Electronic convergence**: If not converged in 60 steps, adjust AMIX = 0.2

## KL Divergence Analysis

### Local Environment Descriptor

For each atomic site, the local environment σ is defined as:

```
σ = {Center sublattice type, (N_Ni neighbors, N_Si neighbors)}
```

with cutoff radius R_cut = 3.2 Å (includes 1st and 2nd coordination shells)

### Calculation

```
D_KL(P || Q) = Σ_σ P(σ) ln(P(σ) / Q(σ))
```

where:
- P(σ): Distribution from target structure
- Q(σ): Baseline distribution from Group A random structures

### Interpretation

- D_KL ≈ 0: Random (disordered) structure
- D_KL > 0: Ordered structure (deviation from random)
- Higher D_KL indicates stronger short-range order

## Usage

### Command Line

```bash
# Run the full workflow with large-scale selection (recommended)
python ni2si_antisite_analysis.py

# Run with Streamlit interface
streamlit run ni2si_antisite_analysis.py -- --streamlit
```

### Python API (Recommended: Large-Scale Selection)

```python
from ni2si_antisite_analysis import run_full_workflow_with_selection

# Run the complete workflow with large-scale candidate generation
# Generates ~15000 candidates and selects top 256 by KL divergence
results = run_full_workflow_with_selection(
    output_dir="project_Ni2Si",
    n_candidates_per_composition=5000,  # 5000 × 3 compositions = 15000 total
    n_select=256,                        # Select top 256 by KL divergence
    pair_potential_k=1.0,
    verbose=True
)

# Access results
selected_configs = results["selected_configurations"]  # Top 256 structures
all_candidates = results["all_candidates"]             # All generated candidates
statistics = results["statistics"]                     # Selection statistics
```

### Customizing Selection

```python
from ni2si_antisite_analysis import run_full_workflow_with_selection

# Specify exact number per composition
results = run_full_workflow_with_selection(
    output_dir="project_Ni2Si",
    n_candidates_per_composition=10000,  # More candidates for better selection
    n_select=256,
    composition_ratio={
        "Ni32Si16": 100,  # 100 stoichiometric
        "Ni34Si14": 78,   # 78 Ni-rich
        "Ni30Si18": 78,   # 78 Si-rich
    },
    verbose=True
)
```

### Legacy API (Group-based)

```python
from ni2si_antisite_analysis import run_full_workflow

# Original workflow: 256 structures (100 random + 100 SA + 56 specific)
results = run_full_workflow(
    output_dir="project_Ni2Si",
    pair_potential_k=1.0,
    verbose=True
)

# Access results
configurations = results["configurations"]
kl_results = results["kl_results"]
```

### Customizing Pair Potential

```python
from ni2si_antisite_analysis import (
    generate_ni2si_supercell,
    build_pair_list_with_weights,
    simulated_annealing,
    SA_CUTOFF,
)

structure = generate_ni2si_supercell()

# Use different exponent k
pair_list = build_pair_list_with_weights(structure, SA_CUTOFF, k=2.0)

# Run SA with custom parameters
final_config, trajectory = simulated_annealing(
    structure=structure,
    n_ni=32,
    n_si=16,
    pair_list=pair_list,
    T_initial=20.0,
    T_final=0.001,
    cooling_rate=0.98,
    steps_per_temp=200,
)
```

## Output Directory Structure

```
project_Ni2Si/
├── 01_structures/
│   ├── site_mapping.json      # Complete site mapping
│   ├── POSCAR_A_000           # Group A structures
│   ├── POSCAR_A_001
│   ├── ...
│   ├── POSCAR_B_000           # Group B structures
│   ├── ...
│   └── POSCAR_C_000           # Group C structures
├── 02_vasp_runs/
│   ├── INCAR_template
│   └── KPOINTS
├── 03_analysis/
│   ├── energy_kl.csv          # Results table
│   └── dos_plots/             # DOS visualizations
└── 04_scripts/
```

### Site Mapping JSON Format

```json
{
  "lattice_vectors": [[10.0, 0, 0], [0, 7.46, 0], [0, 0, 7.04]],
  "supercell_dim": [2, 2, 1],
  "space_group": "Pnma",
  "space_group_number": 62,
  "n_atoms": 48,
  "sites": [
    {
      "index": 0,
      "sublattice": "Ni1",
      "ideal_species": "Ni",
      "frac_coords": [0.019, 0.125, 0.218],
      "cart_coords": [0.19, 0.932, 1.535],
      "image": [0, 0, 0],
      "wyckoff_index": 0
    },
    ...
  ],
  "sublattice_indices": {
    "Ni1": [0, 1, 2, ...],
    "Ni2": [16, 17, 18, ...],
    "Si1": [32, 33, 34, ...]
  }
}
```

## DVM Comparison Analysis

### LOBSTER Integration

For COHP/COOP analysis, generate LOBSTER input after VASP calculation:

```bash
# lobsterin file
COHPstartEnergy -15
COHPendEnergy 5
basisSet pbeVaspFit2015
includeOrbitals s p d
cohpGenerator from 0.1 to 3.5 orbitalwise
```

### Analysis Workflow

1. Run VASP calculations for all structures
2. Run LOBSTER on selected structures (high/low D_KL)
3. Extract ICOHP values for Ni-Si bonds
4. Compare with DVM Bond Overlap Population

### Expected Correlations

- Higher D_KL → Stronger Ni-Si p-d hybridization
- Ordered structures → Pseudo-gap at Fermi level
- Ni1 vs Ni2 antisite → Different destabilization energies

## Troubleshooting

### Convergence Issues

**Electronic convergence failure (> 60 steps)**:
```ini
AMIX = 0.2
BMIX = 0.0001
```

**Magnetic moment collapse**:
- Flag structure for non-magnetic analysis
- Consider ISPIN = 1 calculation for comparison

### Memory Issues

For large-scale calculations:
```ini
NCORE = 8  # Adjust based on cluster
KPAR = 2   # K-point parallelization
```

### Numerical Precision

If KL divergence shows unexpected values:
- Check neighbor list cutoff (should be 3.2 Å)
- Verify PBC distance calculations
- Ensure Laplace smoothing is applied (default: 1e-10)

## References

1. Warren-Cowley SRO parameters for binary alloys
2. Discrete Variational Method (DVM) for electronic structure
3. LOBSTER: Local Orbital Basis Suite Towards Electronic-Structure Reconstruction
4. VASP: Vienna Ab initio Simulation Package

## Contact

For questions about this implementation, please refer to the original instruction document or contact the development team.
