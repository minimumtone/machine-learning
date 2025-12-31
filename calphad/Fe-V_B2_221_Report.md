# Fe-V B2_221 CALPHAD Model Report

## 1. Overview

This report documents the development of a CALPHAD thermodynamic database (TDB) for the Fe-V binary system with a B2_221 ordered phase. The B2_221 phase represents a 2x2x1 supercell expansion of the BCC-B2 structure, resulting in 8 sublattice sites and 256 possible endmember configurations.

### 1.1 Data Source

The formation energies for the 256 endmember configurations were calculated using Density Functional Theory (DFT) and provided in the Excel file `Book1.xlsx`.

![Configuration Distribution](report_figures/01_configuration_distribution.png)

**Figure 1**: Distribution of the 256 configurations by composition. Each bar represents the number of configurations for a given Fe-V composition. Green bars indicate compositions where most configurations converged successfully, while red bars indicate compositions with convergence issues.

## 2. DFT Data Analysis

### 2.1 Convergence Status

Of the 256 DFT calculations performed, 191 (74.6%) converged successfully, while 65 (25.4%) did not converge. The unconverged configurations are primarily concentrated at the composition endpoints (pure Fe and pure V) and at Fe7V1 composition.

![Convergence Status](report_figures/02_convergence_status.png)

**Figure 2**: Convergence status breakdown by composition. The green bars show the number of converged configurations, while the red bars show unconverged configurations for each composition.

### 2.2 Energy vs Composition

The DFT-calculated energy per atom shows a clear trend with composition. A second-order polynomial was fitted to the converged data to extrapolate reference energies for pure Fe and pure V.

![Energy vs Composition](report_figures/03_energy_vs_composition.png)

**Figure 3**: DFT energy per atom as a function of composition. Green points represent converged calculations, red points represent unconverged calculations. The blue dashed line shows the polynomial fit used to extrapolate reference energies. The blue stars mark the extrapolated reference energies for pure Fe (-8.109 eV/atom) and pure V (-9.043 eV/atom).

### 2.3 Polynomial Extrapolation

Since pure Fe (Fe8V0) and pure V (Fe0V8) configurations did not converge, reference energies were extrapolated using a quadratic polynomial fit to the converged data:

```
E(n_V) = 0.001738 * n_V^2 - 0.130649 * n_V - 8.108648  [eV/atom]
```

This yields:
- **E_Fe_ref** = -8.108648 eV/atom (extrapolated for n_V = 0)
- **E_V_ref** = -9.042623 eV/atom (extrapolated for n_V = 8)

## 3. Formation Energy Calculation

### 3.1 Calculation Method

Formation energies were calculated using the standard formula:

```
dH_f = 8 * E_per_atom - n_Fe * E_Fe_ref - n_V * E_V_ref  [eV per 8-atom supercell]
```

The result was then converted to J/mol by multiplying by 96485 J/mol/eV.

For unconverged configurations, the energy per atom was predicted using the composition-averaged values from converged configurations at the same composition, or the polynomial fit for compositions with no converged data.

### 3.2 Formation Energy Distribution

![Formation Energy Analysis](report_figures/04_formation_energy_analysis.png)

**Figure 4**: Formation energy analysis. (a) Formation energy vs composition showing converged (green) and predicted (red) values. (b) Histogram of all formation energies. (c) Box plot showing the distribution of formation energies at each composition. (d) Comparison of formation energy distributions for converged vs predicted configurations.

### 3.3 Formation Energy Statistics

| Parameter | Value |
|-----------|-------|
| Minimum | -140.06 kJ/mol |
| Maximum | 60.14 kJ/mol |
| Mean | -16.56 kJ/mol |
| Std Dev | 33.68 kJ/mol |

Most configurations have negative formation energies, indicating thermodynamic stability relative to the pure element references. The most stable configurations are found at intermediate compositions (Fe4V4 to Fe2V6).

## 4. TDB Structure

### 4.1 B2_221 Phase Model

The B2_221 phase is modeled using an 8-sublattice compound energy formalism (CEF). Each sublattice can be occupied by either Fe or V, resulting in 2^8 = 256 possible endmember configurations.

![B2_221 Structure](report_figures/05_b2_221_structure.png)

**Figure 5**: Schematic of the B2_221 supercell structure. The 8 sublattice sites are divided into corner sites (SL1-4, blue) and body-center sites (SL5-8, red). Each site can be occupied by either Fe or V.

### 4.2 TDB File Organization

![TDB Structure Workflow](report_figures/06_tdb_structure_workflow.png)

**Figure 6**: Data flow and TDB file structure. DFT calculations provide formation energies, which are processed and stored in GFxVyyy functions. The PARAMETER G formula combines temperature-dependent terms (DHFE, DHV) with the DFT formation energies.

### 4.3 Key TDB Components

1. **GHSER Functions**: Standard SGTE reference functions for Fe and V
   - GHSERFE: Gibbs energy of BCC Fe relative to SER
   - GHSERV: Gibbs energy of BCC V relative to SER

2. **DHFE/DHV Functions**: Temperature-dependent terms shifted to be zero at 298.15K
   - DHFE = GHSERFE - GHSERFE(298.15K)
   - DHV = GHSERV - GHSERV(298.15K)

3. **GFxVyyy Functions**: 256 formation energy functions (one per endmember)
   - Naming convention: GF{n_V}V{sequence_number}
   - Example: GF4V001 = formation energy for the first Fe4V4 configuration

4. **PARAMETER G**: Gibbs energy of each endmember
   ```
   G(B2_221, SL1:SL2:...:SL8) = n_Fe*DHFE + n_V*DHV + GFxVyyy
   ```

### 4.4 298.15K Reference State

The DHFE and DHV functions are designed to be zero at 298.15K. This approach was chosen because:

1. DFT calculations provide formation energies at 0K
2. GHSER functions contain arbitrary constant terms for low-temperature fitting
3. Using 298.15K as the reference avoids issues with the low-temperature constant terms
4. The temperature dependence (entropy, heat capacity) is captured by the GHSER functions

## 5. Energy Comparison

![Energy Comparison](report_figures/07_energy_comparison.png)

**Figure 7**: (a) Energy spread by composition showing the variation in DFT energies for each composition. Converged calculations (green circles) generally show less scatter than unconverged calculations (red X). (b) Average energy by composition for converged calculations only, with error bars showing standard deviation.

## 6. Summary Statistics

![Summary Statistics](report_figures/08_summary_statistics.png)

**Figure 8**: Summary of key parameters and statistics for the Fe-V B2_221 CALPHAD model.

## 7. Phase Diagram Calculation Tools

Two tools have been created for calculating the metastable BCC phase diagram:

### 7.1 pycalphad Script

The Python script `fev_bcc_pycalphad.py` uses the pycalphad library to calculate the phase diagram considering only BCC_A2 and B2_221 phases.

Usage:
```bash
python calphad/fev_bcc_pycalphad.py
```

### 7.2 Thermo-Calc Macro

The macro file `fev_bcc_thermocalc.tcm` can be used in Thermo-Calc Console to calculate the same metastable BCC phase diagram.

Usage in Thermo-Calc:
```
MACRO_FILE_READ fev_bcc_thermocalc.tcm
```

## 8. Files Summary

| File | Description |
|------|-------------|
| `Fe-V_B2_221.tdb` | CALPHAD thermodynamic database |
| `fev_excel_mapping.csv` | Mapping between Excel config_index and TDB function names |
| `fev_bcc_pycalphad.py` | pycalphad script for BCC phase diagram |
| `fev_bcc_thermocalc.tcm` | Thermo-Calc macro for BCC phase diagram |
| `report_figures/` | Directory containing all report figures |

## 9. Notes and Limitations

1. **Unconverged Data**: 65 of 256 configurations did not converge in DFT. These were predicted using polynomial extrapolation or composition-averaged values.

2. **Reference Energies**: Pure Fe and pure V reference energies were extrapolated, not directly calculated.

3. **Interaction Parameters**: All L parameters in the TDB are set to zero (no mixing interactions beyond the endmember energies).

4. **Temperature Range**: The model is designed for calculations in the 1000K-2500K range where the A2/B2 ordering transition is expected.

---

*Report generated: 2025-12-31*
*Data source: Book1.xlsx (Fe-V DFT calculations)*
*PR: https://github.com/minimumtone/machine-learning/pull/73*
