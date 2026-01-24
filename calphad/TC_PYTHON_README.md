# TC-Python Scripts for Fe-V System

This directory contains TC-Python scripts for thermodynamic calculations of the Fe-V binary system using the B2_221 ordered phase model.

## Overview

TC-Python is Thermo-Calc's official Python API that enables programmatic access to thermodynamic calculations. These scripts provide functionality equivalent to the existing pycalphad script (`fev_bcc_pycalphad.py`) and Thermo-Calc macro (`fev_bcc_thermocalc.tcm`), but using TC-Python for more advanced features and better integration with Thermo-Calc's calculation engine.

## Requirements

1. **Thermo-Calc Installation**: A valid Thermo-Calc installation with an active license
2. **TC-Python SDK**: TC-Python must be installed and configured
3. **Python Dependencies**: numpy, pandas, matplotlib
4. **Database File**: `Fe-V_B2_221.tdb` (included in this directory)

### Installing TC-Python

TC-Python is included with Thermo-Calc installations. To set it up:

1. Locate your Thermo-Calc installation directory
2. Run the TC-Python installer from the installation directory
3. Verify installation by running:
   ```python
   from tc_python import TCPython
   print("TC-Python installed successfully")
   ```

For detailed instructions, visit: https://thermocalc.com/products/software-development-kits/tc-python/

## Scripts

### 1. fev_bcc_tcpython.py

Main script for calculating the metastable BCC phase diagram.

**Features:**
- Load custom TDB database (Fe-V_B2_221.tdb)
- Calculate metastable BCC phase diagram (BCC_A2 + B2_221 only)
- Point-by-point equilibrium calculations
- Gibbs energy surface calculation
- Generate phase diagram plots

**Usage:**
```bash
python fev_bcc_tcpython.py
```

**Output:**
- `Fe-V_BCC_phase_diagram_tcpython.png`: Phase diagram plot
- `Fe-V_BCC_phase_data_tcpython.csv`: Phase boundary data
- `Fe-V_Gibbs_energy_tcpython.png`: Gibbs energy surface at 1000 K

### 2. fev_property_diagram_tcpython.py

Script for calculating thermodynamic property diagrams.

**Features:**
- Property vs temperature diagrams (G, H, S, Cp)
- Property vs composition diagrams
- Phase fraction vs temperature
- Comparison between BCC_A2 and B2_221 phases

**Usage:**
```bash
python fev_property_diagram_tcpython.py
```

**Output:**
- `Fe-V_property_vs_T_xV0.50.png`: Properties vs temperature at x(V)=0.5
- `Fe-V_phase_fraction_vs_T.png`: Phase fractions vs temperature
- `Fe-V_property_vs_xV_T1000K.png`: Properties vs composition at 1000 K

## Comparison with Other Tools

| Feature | pycalphad | TC-Python | TCM Macro |
|---------|-----------|-----------|-----------|
| Open source | Yes | No | No |
| License required | No | Yes | Yes |
| Phase diagram mapping | Limited | Full | Full |
| Property diagrams | Manual | Built-in | Built-in |
| Diffusion calculations | No | Yes | Yes |
| Scheil solidification | No | Yes | Yes |
| Custom TDB support | Yes | Yes | Yes |

## TC-Python API Overview

### Basic Workflow

```python
from tc_python import TCPython, ThermodynamicQuantity

with TCPython() as tc:
    # Load user database
    system = (tc
              .select_user_database_and_elements("Fe-V_B2_221.tdb", ["FE", "V"])
              .without_default_phases()
              .select_phase("BCC_A2")
              .select_phase("B2_221")
              .get_system())
    
    # Single equilibrium calculation
    calc = (system
            .with_single_equilibrium_calculation()
            .set_condition(ThermodynamicQuantity.temperature(), 1000)
            .set_condition(ThermodynamicQuantity.pressure(), 101325)
            .set_condition(
                ThermodynamicQuantity.mole_fraction_of_a_component("V"), 0.5))
    
    result = calc.calculate()
    
    # Get results
    G = result.get_value_of('GM')
    stable_phases = result.get_stable_phases()
```

### Key Methods

- `select_user_database_and_elements()`: Load custom TDB file
- `without_default_phases()`: Clear default phase selection
- `select_phase()`: Add phase to calculation
- `with_single_equilibrium_calculation()`: Single point calculation
- `with_phase_diagram_calculation()`: Phase diagram mapping
- `set_condition()`: Set calculation conditions (T, P, composition)
- `calculate()`: Execute calculation
- `get_value_of()`: Extract results (GM, HM, SM, NP, etc.)
- `get_stable_phases()`: Get list of stable phases

### Available Thermodynamic Quantities

- `GM`: Molar Gibbs energy (J/mol)
- `HM`: Molar enthalpy (J/mol)
- `SM`: Molar entropy (J/mol/K)
- `CPM`: Molar heat capacity (J/mol/K)
- `NP(phase)`: Phase fraction
- `X(phase,element)`: Composition in phase
- `Y(phase,sublattice,species)`: Site fraction

## Database Information

The `Fe-V_B2_221.tdb` database contains:

- **Elements**: Fe, V
- **Phases**:
  - LIQUID
  - FCC_A1
  - BCC_A2 (disordered BCC)
  - B2_221 (8-sublattice ordered B2)
  - SIGMA

- **B2_221 Model**: 8 sublattices (2x2x1 supercell) with 256 endmembers
- **Formation energies**: From DFT calculations
- **Temperature dependence**: Neumann-Kopp approximation (CALPHAD-NK)

For detailed information about the database, see `Fe-V_B2_221_Report.md`.

## Troubleshooting

### TC-Python not found
```
ImportError: No module named 'tc_python'
```
Solution: Ensure TC-Python is installed and the Python environment is correctly configured.

### License error
```
Error: No valid Thermo-Calc license found
```
Solution: Check that your Thermo-Calc license is active and properly configured.

### TDB file not found
```
Error: TDB file not found: Fe-V_B2_221.tdb
```
Solution: Ensure the TDB file is in the same directory as the script.

## References

1. Thermo-Calc TC-Python Documentation: https://thermocalc.com/products/software-development-kits/tc-python/
2. Thermo-Calc Learning Hub: https://learn.thermocalc.com/courses/introduction-to-tc-python/
3. Fe-V B2_221 CALPHAD Model: See `Fe-V_B2_221_Report.md`
4. K.C. Hari Kumar and V. Raghavan, CALPHAD Vol. 15, No. 3, pp. 307-314, 1991

## Author

Created for the Fe-V CALPHAD project.
Contact: satoshi minamoto
