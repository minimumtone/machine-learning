#!/usr/bin/env python3
"""
Fe-V BCC Phase Diagram Calculation using TC-Python

This script calculates the metastable BCC phase diagram for the Fe-V system
using TC-Python (Thermo-Calc Python API), considering only BCC_A2 and B2_221
phases (excluding FCC_A1 and SIGMA).

Requirements:
    - Thermo-Calc installation with valid license
    - TC-Python SDK installed
    - Fe-V_B2_221.tdb database file

Usage:
    python fev_bcc_tcpython.py

Output:
    - Fe-V_BCC_phase_diagram_tcpython.png: Phase diagram plot
    - Fe-V_BCC_phase_data_tcpython.csv: Phase boundary data

Note:
    This script requires a valid Thermo-Calc license and TC-Python installation.
    If TC-Python is not available, the script will provide instructions for
    installation and exit gracefully.

Reference:
    - Thermo-Calc TC-Python documentation
    - Fe-V_B2_221_Report.md for CALPHAD model details
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TDB_FILE = os.path.join(SCRIPT_DIR, 'Fe-V_B2_221.tdb')

# Try to import TC-Python
try:
    from tc_python import TCPython, ThermodynamicQuantity, SetUp
    TC_PYTHON_AVAILABLE = True
except ImportError:
    TC_PYTHON_AVAILABLE = False
    print("Warning: TC-Python is not installed.")
    print("To use this script, please install TC-Python from Thermo-Calc.")
    print("Visit: https://thermocalc.com/products/software-development-kits/tc-python/")


def check_tdb_file() -> bool:
    """Check if the TDB file exists."""
    if not os.path.exists(TDB_FILE):
        print(f"Error: TDB file not found: {TDB_FILE}")
        return False
    print(f"TDB file found: {TDB_FILE}")
    return True


def calculate_single_equilibrium(
    tc_session,
    temperature: float,
    x_v: float,
    phases: list
) -> dict:
    """
    Calculate single equilibrium at given temperature and composition.

    Parameters
    ----------
    tc_session : TCPython session
        Active TC-Python session
    temperature : float
        Temperature in K
    x_v : float
        Mole fraction of V
    phases : list
        List of phases to consider

    Returns
    -------
    dict
        Dictionary containing equilibrium results
    """
    try:
        # Create single equilibrium calculation
        calc = (tc_session
                .with_single_equilibrium_calculation()
                .set_condition(ThermodynamicQuantity.temperature(), temperature)
                .set_condition(ThermodynamicQuantity.pressure(), 101325)
                .set_condition(
                    ThermodynamicQuantity.mole_fraction_of_a_component("V"),
                    max(x_v, 1e-10)
                ))

        # Calculate equilibrium
        result = calc.calculate()

        # Get stable phases and their fractions
        stable_phases = result.get_stable_phases()
        phase_fractions = {}
        for phase in stable_phases:
            try:
                nf = result.get_value_of(f'NP({phase})')
                if nf > 0.01:
                    phase_fractions[phase] = nf
            except Exception:
                pass

        return {
            'T': temperature,
            'x_V': x_v,
            'stable_phases': list(phase_fractions.keys()),
            'phase_fractions': phase_fractions,
            'success': True
        }

    except Exception as e:
        return {
            'T': temperature,
            'x_V': x_v,
            'stable_phases': [],
            'phase_fractions': {},
            'success': False,
            'error': str(e)
        }


def calculate_bcc_phase_diagram_tcpython(
    T_min: float = 800,
    T_max: float = 2000,
    T_step: float = 50,
    x_V_min: float = 0.0,
    x_V_max: float = 1.0,
    x_V_step: float = 0.05
) -> Optional[dict]:
    """
    Calculate the metastable BCC phase diagram using TC-Python.

    Parameters
    ----------
    T_min, T_max : float
        Temperature range in K
    T_step : float
        Temperature step in K
    x_V_min, x_V_max : float
        V composition range (mole fraction)
    x_V_step : float
        Composition step

    Returns
    -------
    dict or None
        Dictionary containing phase diagram data, or None if calculation fails
    """
    if not TC_PYTHON_AVAILABLE:
        print("TC-Python is not available. Cannot perform calculation.")
        return None

    if not check_tdb_file():
        return None

    print("=" * 60)
    print("Fe-V BCC Phase Diagram Calculation using TC-Python")
    print("=" * 60)

    # Temperature and composition grids
    temperatures = np.arange(T_min, T_max + T_step, T_step)
    compositions = np.arange(x_V_min, x_V_max + x_V_step, x_V_step)

    print(f"Temperature range: {T_min} - {T_max} K (step: {T_step} K)")
    print(f"Composition range: {x_V_min} - {x_V_max} (step: {x_V_step})")
    print(f"Total calculation points: {len(temperatures) * len(compositions)}")

    # BCC phases only (metastable diagram)
    bcc_phases = ['BCC_A2', 'B2_221']

    # Store results
    results = {
        'T': [],
        'x_V': [],
        'phases': [],
        'phase_fractions': []
    }

    try:
        with TCPython() as tc:
            # Load user database (TDB file)
            print(f"\nLoading database: {TDB_FILE}")

            # Set up the system with user database
            system = (tc
                      .select_user_database_and_elements(TDB_FILE, ["FE", "V"])
                      .without_default_phases()
                      .select_phase("BCC_A2")
                      .select_phase("B2_221")
                      .get_system())

            print(f"System configured with phases: {bcc_phases}")
            print("\nCalculating equilibrium points...")

            # Calculate equilibrium at each point
            total_points = len(temperatures) * len(compositions)
            point_count = 0

            for T in temperatures:
                for x_V in compositions:
                    point_count += 1
                    if point_count % 50 == 0:
                        print(f"  Progress: {point_count}/{total_points} "
                              f"({100*point_count/total_points:.1f}%)")

                    try:
                        # Create single equilibrium calculation
                        calc = (system
                                .with_single_equilibrium_calculation()
                                .set_condition(
                                    ThermodynamicQuantity.temperature(), T)
                                .set_condition(
                                    ThermodynamicQuantity.pressure(), 101325)
                                .set_condition(
                                    ThermodynamicQuantity.mole_fraction_of_a_component("V"),
                                    max(x_V, 1e-10)))

                        # Calculate equilibrium
                        result = calc.calculate()

                        # Get stable phases
                        stable_phases = result.get_stable_phases()
                        phase_names = []
                        phase_fracs = []

                        for phase in stable_phases:
                            if phase in bcc_phases:
                                try:
                                    nf = result.get_value_of(f'NP({phase})')
                                    if nf > 0.01:
                                        phase_names.append(phase)
                                        phase_fracs.append(nf)
                                except Exception:
                                    pass

                        results['T'].append(T)
                        results['x_V'].append(x_V)
                        results['phases'].append(
                            '+'.join(phase_names) if phase_names else 'unknown')
                        results['phase_fractions'].append(phase_fracs)

                    except Exception as e:
                        results['T'].append(T)
                        results['x_V'].append(x_V)
                        results['phases'].append('error')
                        results['phase_fractions'].append([])

            print(f"\nCalculation completed: {point_count} points")
            return results

    except Exception as e:
        print(f"Error during TC-Python calculation: {e}")
        return None


def calculate_phase_diagram_mapping(
    T_min: float = 800,
    T_max: float = 2000,
    x_V_min: float = 0.0,
    x_V_max: float = 1.0
) -> Optional[dict]:
    """
    Calculate phase diagram using TC-Python's phase diagram mapping.

    This method uses TC-Python's built-in phase diagram calculation
    which is more efficient than point-by-point equilibrium calculations.

    Parameters
    ----------
    T_min, T_max : float
        Temperature range in K
    x_V_min, x_V_max : float
        V composition range (mole fraction)

    Returns
    -------
    dict or None
        Dictionary containing phase diagram data
    """
    if not TC_PYTHON_AVAILABLE:
        print("TC-Python is not available.")
        return None

    if not check_tdb_file():
        return None

    print("=" * 60)
    print("Fe-V Phase Diagram Mapping using TC-Python")
    print("=" * 60)

    try:
        with TCPython() as tc:
            # Load user database
            print(f"Loading database: {TDB_FILE}")

            # Set up the system
            system = (tc
                      .select_user_database_and_elements(TDB_FILE, ["FE", "V"])
                      .without_default_phases()
                      .select_phase("BCC_A2")
                      .select_phase("B2_221")
                      .get_system())

            # Create phase diagram calculation
            print("Setting up phase diagram calculation...")
            phase_diagram = (system
                             .with_phase_diagram_calculation()
                             .with_first_axis(
                                 ThermodynamicQuantity.mole_fraction_of_a_component("V"),
                                 x_V_min if x_V_min > 0 else 1e-6,
                                 x_V_max)
                             .with_second_axis(
                                 ThermodynamicQuantity.temperature(),
                                 T_min,
                                 T_max)
                             .set_condition(
                                 ThermodynamicQuantity.pressure(), 101325))

            # Calculate
            print("Calculating phase diagram...")
            result = phase_diagram.calculate()

            # Extract phase boundaries
            print("Extracting phase boundaries...")
            phase_regions = []

            # Get all phase regions from the result
            for region in result.get_phase_regions():
                region_data = {
                    'phases': region.get_stable_phases(),
                    'boundaries': []
                }
                for boundary in region.get_boundaries():
                    boundary_points = boundary.get_boundary_points()
                    region_data['boundaries'].append(boundary_points)
                phase_regions.append(region_data)

            return {
                'phase_regions': phase_regions,
                'T_range': (T_min, T_max),
                'x_V_range': (x_V_min, x_V_max)
            }

    except Exception as e:
        print(f"Error during phase diagram mapping: {e}")
        print("Falling back to point-by-point calculation...")
        return None


def plot_phase_diagram(
    results: dict,
    output_file: str = 'Fe-V_BCC_phase_diagram_tcpython.png'
) -> None:
    """
    Plot the BCC phase diagram.

    Parameters
    ----------
    results : dict
        Phase diagram calculation results
    output_file : str
        Output file path
    """
    if results is None:
        print("No results to plot")
        return

    # Create DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define colors for different phase regions
    phase_colors = {
        'BCC_A2': 'blue',
        'B2_221': 'red',
        'BCC_A2+B2_221': 'purple',
        'B2_221+BCC_A2': 'purple',
        'unknown': 'gray',
        'error': 'black'
    }

    # Plot each phase region
    for phase, color in phase_colors.items():
        mask = df['phases'] == phase
        if mask.any():
            ax.scatter(df.loc[mask, 'x_V'], df.loc[mask, 'T'],
                       c=color, s=5, label=phase, alpha=0.7)

    # Labels and title
    ax.set_xlabel('Mole Fraction V', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title('Fe-V Metastable BCC Phase Diagram (TC-Python)\n'
                 '(BCC_A2 + B2_221 only)', fontsize=14)
    ax.legend(loc='best')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # Save figure
    output_path = os.path.join(SCRIPT_DIR, output_file)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved phase diagram to: {output_path}")
    plt.close()


def save_phase_data(
    results: dict,
    output_file: str = 'Fe-V_BCC_phase_data_tcpython.csv'
) -> None:
    """Save phase diagram data to CSV."""
    if results is None:
        return

    df = pd.DataFrame({
        'T_K': results['T'],
        'x_V': results['x_V'],
        'phases': results['phases']
    })

    output_path = os.path.join(SCRIPT_DIR, output_file)
    df.to_csv(output_path, index=False)
    print(f"Saved phase data to: {output_path}")


def calculate_gibbs_energy_surface(
    temperature: float = 1000,
    x_V_min: float = 0.0,
    x_V_max: float = 1.0,
    x_V_step: float = 0.02
) -> Optional[dict]:
    """
    Calculate Gibbs energy surface for BCC phases at a given temperature.

    Parameters
    ----------
    temperature : float
        Temperature in K
    x_V_min, x_V_max : float
        V composition range
    x_V_step : float
        Composition step

    Returns
    -------
    dict or None
        Dictionary containing Gibbs energy data
    """
    if not TC_PYTHON_AVAILABLE:
        print("TC-Python is not available.")
        return None

    if not check_tdb_file():
        return None

    print(f"\nCalculating Gibbs energy surface at T = {temperature} K")

    compositions = np.arange(x_V_min + 0.001, x_V_max, x_V_step)

    results = {
        'x_V': [],
        'G_BCC_A2': [],
        'G_B2_221': [],
        'G_equilibrium': []
    }

    try:
        with TCPython() as tc:
            # Set up the system
            system = (tc
                      .select_user_database_and_elements(TDB_FILE, ["FE", "V"])
                      .without_default_phases()
                      .select_phase("BCC_A2")
                      .select_phase("B2_221")
                      .get_system())

            for x_V in compositions:
                try:
                    # Calculate equilibrium
                    calc = (system
                            .with_single_equilibrium_calculation()
                            .set_condition(
                                ThermodynamicQuantity.temperature(), temperature)
                            .set_condition(
                                ThermodynamicQuantity.pressure(), 101325)
                            .set_condition(
                                ThermodynamicQuantity.mole_fraction_of_a_component("V"),
                                x_V))

                    result = calc.calculate()

                    # Get Gibbs energies
                    G_eq = result.get_value_of('GM')

                    # Try to get individual phase Gibbs energies
                    try:
                        G_bcc = result.get_value_of('GM(BCC_A2)')
                    except Exception:
                        G_bcc = np.nan

                    try:
                        G_b2 = result.get_value_of('GM(B2_221)')
                    except Exception:
                        G_b2 = np.nan

                    results['x_V'].append(x_V)
                    results['G_BCC_A2'].append(G_bcc)
                    results['G_B2_221'].append(G_b2)
                    results['G_equilibrium'].append(G_eq)

                except Exception:
                    results['x_V'].append(x_V)
                    results['G_BCC_A2'].append(np.nan)
                    results['G_B2_221'].append(np.nan)
                    results['G_equilibrium'].append(np.nan)

            return results

    except Exception as e:
        print(f"Error calculating Gibbs energy: {e}")
        return None


def plot_gibbs_energy_surface(
    results: dict,
    temperature: float,
    output_file: str = 'Fe-V_Gibbs_energy_tcpython.png'
) -> None:
    """Plot Gibbs energy surface."""
    if results is None:
        print("No results to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x_V = np.array(results['x_V'])
    G_bcc = np.array(results['G_BCC_A2'])
    G_b2 = np.array(results['G_B2_221'])
    G_eq = np.array(results['G_equilibrium'])

    # Plot individual phase energies
    mask_bcc = ~np.isnan(G_bcc)
    mask_b2 = ~np.isnan(G_b2)

    if mask_bcc.any():
        ax.plot(x_V[mask_bcc], G_bcc[mask_bcc], 'b-', label='BCC_A2', linewidth=2)
    if mask_b2.any():
        ax.plot(x_V[mask_b2], G_b2[mask_b2], 'r-', label='B2_221', linewidth=2)

    # Plot equilibrium energy
    ax.plot(x_V, G_eq, 'k--', label='Equilibrium', linewidth=1.5)

    ax.set_xlabel('Mole Fraction V', fontsize=12)
    ax.set_ylabel('Gibbs Energy (J/mol)', fontsize=12)
    ax.set_title(f'Fe-V Gibbs Energy Surface at T = {temperature} K\n'
                 '(TC-Python)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(SCRIPT_DIR, output_file)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved Gibbs energy plot to: {output_path}")
    plt.close()


def main():
    """Main function."""
    print("=" * 60)
    print("Fe-V BCC Phase Diagram Calculation using TC-Python")
    print("=" * 60)

    if not TC_PYTHON_AVAILABLE:
        print("\nTC-Python is not installed. Exiting.")
        print("\nTo install TC-Python:")
        print("1. Ensure you have a valid Thermo-Calc license")
        print("2. Install TC-Python from Thermo-Calc installation")
        print("3. Follow instructions at:")
        print("   https://thermocalc.com/products/software-development-kits/tc-python/")
        sys.exit(1)

    # Check TDB file
    if not check_tdb_file():
        sys.exit(1)

    # Calculate phase diagram using point-by-point method
    print("\nCalculating metastable BCC phase diagram...")
    results = calculate_bcc_phase_diagram_tcpython(
        T_min=800,
        T_max=2000,
        T_step=50,
        x_V_min=0.0,
        x_V_max=1.0,
        x_V_step=0.05
    )

    # Plot and save results
    if results:
        print("\nGenerating plots...")
        plot_phase_diagram(results)
        save_phase_data(results)

        # Calculate Gibbs energy surface at 1000 K
        print("\nCalculating Gibbs energy surface at 1000 K...")
        gibbs_results = calculate_gibbs_energy_surface(temperature=1000)
        if gibbs_results:
            plot_gibbs_energy_surface(gibbs_results, temperature=1000)

    print("\nDone!")


if __name__ == '__main__':
    main()
