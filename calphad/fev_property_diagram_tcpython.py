#!/usr/bin/env python3
"""
Fe-V Property Diagram Calculation using TC-Python

This script calculates various thermodynamic properties as a function of
temperature or composition for the Fe-V system using TC-Python.

Features:
    - Property diagrams (Gibbs energy, enthalpy, entropy vs T or composition)
    - Phase fraction vs temperature
    - Sublattice site fractions for B2_221 ordered phase
    - Comparison between BCC_A2 and B2_221 phases

Requirements:
    - Thermo-Calc installation with valid license
    - TC-Python SDK installed
    - Fe-V_B2_221.tdb database file

Usage:
    python fev_property_diagram_tcpython.py

Reference:
    - Thermo-Calc TC-Python documentation
    - Fe-V_B2_221_Report.md for CALPHAD model details
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TDB_FILE = os.path.join(SCRIPT_DIR, 'Fe-V_B2_221.tdb')

# Try to import TC-Python
try:
    from tc_python import TCPython, ThermodynamicQuantity
    TC_PYTHON_AVAILABLE = True
except ImportError:
    TC_PYTHON_AVAILABLE = False
    print("Warning: TC-Python is not installed.")


def check_tdb_file() -> bool:
    """Check if the TDB file exists."""
    if not os.path.exists(TDB_FILE):
        print(f"Error: TDB file not found: {TDB_FILE}")
        return False
    return True


def calculate_property_vs_temperature(
    x_v: float = 0.5,
    T_min: float = 300,
    T_max: float = 2000,
    T_step: float = 25,
    properties: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Calculate thermodynamic properties as a function of temperature.

    Parameters
    ----------
    x_v : float
        Mole fraction of V
    T_min, T_max : float
        Temperature range in K
    T_step : float
        Temperature step in K
    properties : list, optional
        List of properties to calculate. Default: ['GM', 'HM', 'SM', 'CPM']

    Returns
    -------
    dict or None
        Dictionary containing property data
    """
    if not TC_PYTHON_AVAILABLE:
        print("TC-Python is not available.")
        return None

    if not check_tdb_file():
        return None

    if properties is None:
        properties = ['GM', 'HM', 'SM', 'CPM']

    print(f"\nCalculating properties vs temperature at x(V) = {x_v}")
    print(f"Temperature range: {T_min} - {T_max} K")

    temperatures = np.arange(T_min, T_max + T_step, T_step)

    results = {
        'T': [],
        'stable_phases': []
    }
    for prop in properties:
        results[prop] = []
        results[f'{prop}_BCC_A2'] = []
        results[f'{prop}_B2_221'] = []

    try:
        with TCPython() as tc:
            # Set up the system
            system = (tc
                      .select_user_database_and_elements(TDB_FILE, ["FE", "V"])
                      .without_default_phases()
                      .select_phase("BCC_A2")
                      .select_phase("B2_221")
                      .get_system())

            for T in temperatures:
                try:
                    calc = (system
                            .with_single_equilibrium_calculation()
                            .set_condition(ThermodynamicQuantity.temperature(), T)
                            .set_condition(ThermodynamicQuantity.pressure(), 101325)
                            .set_condition(
                                ThermodynamicQuantity.mole_fraction_of_a_component("V"),
                                max(x_v, 1e-10)))

                    result = calc.calculate()

                    results['T'].append(T)
                    results['stable_phases'].append(
                        '+'.join(result.get_stable_phases()))

                    for prop in properties:
                        # Equilibrium property
                        try:
                            val = result.get_value_of(prop)
                            results[prop].append(val)
                        except Exception:
                            results[prop].append(np.nan)

                        # BCC_A2 property
                        try:
                            val = result.get_value_of(f'{prop}(BCC_A2)')
                            results[f'{prop}_BCC_A2'].append(val)
                        except Exception:
                            results[f'{prop}_BCC_A2'].append(np.nan)

                        # B2_221 property
                        try:
                            val = result.get_value_of(f'{prop}(B2_221)')
                            results[f'{prop}_B2_221'].append(val)
                        except Exception:
                            results[f'{prop}_B2_221'].append(np.nan)

                except Exception as e:
                    results['T'].append(T)
                    results['stable_phases'].append('error')
                    for prop in properties:
                        results[prop].append(np.nan)
                        results[f'{prop}_BCC_A2'].append(np.nan)
                        results[f'{prop}_B2_221'].append(np.nan)

            return results

    except Exception as e:
        print(f"Error: {e}")
        return None


def calculate_phase_fraction_vs_temperature(
    x_v: float = 0.5,
    T_min: float = 300,
    T_max: float = 2000,
    T_step: float = 25
) -> Optional[Dict[str, Any]]:
    """
    Calculate phase fractions as a function of temperature.

    Parameters
    ----------
    x_v : float
        Mole fraction of V
    T_min, T_max : float
        Temperature range in K
    T_step : float
        Temperature step in K

    Returns
    -------
    dict or None
        Dictionary containing phase fraction data
    """
    if not TC_PYTHON_AVAILABLE:
        print("TC-Python is not available.")
        return None

    if not check_tdb_file():
        return None

    print(f"\nCalculating phase fractions vs temperature at x(V) = {x_v}")

    temperatures = np.arange(T_min, T_max + T_step, T_step)

    results = {
        'T': [],
        'NP_BCC_A2': [],
        'NP_B2_221': []
    }

    try:
        with TCPython() as tc:
            system = (tc
                      .select_user_database_and_elements(TDB_FILE, ["FE", "V"])
                      .without_default_phases()
                      .select_phase("BCC_A2")
                      .select_phase("B2_221")
                      .get_system())

            for T in temperatures:
                try:
                    calc = (system
                            .with_single_equilibrium_calculation()
                            .set_condition(ThermodynamicQuantity.temperature(), T)
                            .set_condition(ThermodynamicQuantity.pressure(), 101325)
                            .set_condition(
                                ThermodynamicQuantity.mole_fraction_of_a_component("V"),
                                max(x_v, 1e-10)))

                    result = calc.calculate()

                    results['T'].append(T)

                    try:
                        np_bcc = result.get_value_of('NP(BCC_A2)')
                        results['NP_BCC_A2'].append(np_bcc)
                    except Exception:
                        results['NP_BCC_A2'].append(0.0)

                    try:
                        np_b2 = result.get_value_of('NP(B2_221)')
                        results['NP_B2_221'].append(np_b2)
                    except Exception:
                        results['NP_B2_221'].append(0.0)

                except Exception:
                    results['T'].append(T)
                    results['NP_BCC_A2'].append(np.nan)
                    results['NP_B2_221'].append(np.nan)

            return results

    except Exception as e:
        print(f"Error: {e}")
        return None


def calculate_property_vs_composition(
    temperature: float = 1000,
    x_V_min: float = 0.01,
    x_V_max: float = 0.99,
    x_V_step: float = 0.02,
    properties: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Calculate thermodynamic properties as a function of composition.

    Parameters
    ----------
    temperature : float
        Temperature in K
    x_V_min, x_V_max : float
        V composition range
    x_V_step : float
        Composition step
    properties : list, optional
        List of properties to calculate

    Returns
    -------
    dict or None
        Dictionary containing property data
    """
    if not TC_PYTHON_AVAILABLE:
        print("TC-Python is not available.")
        return None

    if not check_tdb_file():
        return None

    if properties is None:
        properties = ['GM', 'HM', 'SM']

    print(f"\nCalculating properties vs composition at T = {temperature} K")

    compositions = np.arange(x_V_min, x_V_max + x_V_step, x_V_step)

    results = {
        'x_V': [],
        'stable_phases': []
    }
    for prop in properties:
        results[prop] = []
        results[f'{prop}_BCC_A2'] = []
        results[f'{prop}_B2_221'] = []

    try:
        with TCPython() as tc:
            system = (tc
                      .select_user_database_and_elements(TDB_FILE, ["FE", "V"])
                      .without_default_phases()
                      .select_phase("BCC_A2")
                      .select_phase("B2_221")
                      .get_system())

            for x_V in compositions:
                try:
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

                    results['x_V'].append(x_V)
                    results['stable_phases'].append(
                        '+'.join(result.get_stable_phases()))

                    for prop in properties:
                        try:
                            val = result.get_value_of(prop)
                            results[prop].append(val)
                        except Exception:
                            results[prop].append(np.nan)

                        try:
                            val = result.get_value_of(f'{prop}(BCC_A2)')
                            results[f'{prop}_BCC_A2'].append(val)
                        except Exception:
                            results[f'{prop}_BCC_A2'].append(np.nan)

                        try:
                            val = result.get_value_of(f'{prop}(B2_221)')
                            results[f'{prop}_B2_221'].append(val)
                        except Exception:
                            results[f'{prop}_B2_221'].append(np.nan)

                except Exception:
                    results['x_V'].append(x_V)
                    results['stable_phases'].append('error')
                    for prop in properties:
                        results[prop].append(np.nan)
                        results[f'{prop}_BCC_A2'].append(np.nan)
                        results[f'{prop}_B2_221'].append(np.nan)

            return results

    except Exception as e:
        print(f"Error: {e}")
        return None


def plot_property_vs_temperature(
    results: Dict[str, Any],
    x_v: float,
    output_prefix: str = 'Fe-V_property_vs_T'
) -> None:
    """Plot thermodynamic properties vs temperature."""
    if results is None:
        return

    T = np.array(results['T'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Gibbs energy
    ax = axes[0, 0]
    if 'GM' in results:
        ax.plot(T, results['GM'], 'k-', label='Equilibrium', linewidth=2)
    if 'GM_BCC_A2' in results:
        mask = ~np.isnan(results['GM_BCC_A2'])
        if np.any(mask):
            ax.plot(np.array(T)[mask], np.array(results['GM_BCC_A2'])[mask],
                    'b--', label='BCC_A2', linewidth=1.5)
    if 'GM_B2_221' in results:
        mask = ~np.isnan(results['GM_B2_221'])
        if np.any(mask):
            ax.plot(np.array(T)[mask], np.array(results['GM_B2_221'])[mask],
                    'r--', label='B2_221', linewidth=1.5)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Gibbs Energy (J/mol)')
    ax.set_title(f'Gibbs Energy vs Temperature\nx(V) = {x_v}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Enthalpy
    ax = axes[0, 1]
    if 'HM' in results:
        ax.plot(T, results['HM'], 'k-', label='Equilibrium', linewidth=2)
    if 'HM_BCC_A2' in results:
        mask = ~np.isnan(results['HM_BCC_A2'])
        if np.any(mask):
            ax.plot(np.array(T)[mask], np.array(results['HM_BCC_A2'])[mask],
                    'b--', label='BCC_A2', linewidth=1.5)
    if 'HM_B2_221' in results:
        mask = ~np.isnan(results['HM_B2_221'])
        if np.any(mask):
            ax.plot(np.array(T)[mask], np.array(results['HM_B2_221'])[mask],
                    'r--', label='B2_221', linewidth=1.5)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Enthalpy (J/mol)')
    ax.set_title(f'Enthalpy vs Temperature\nx(V) = {x_v}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 0]
    if 'SM' in results:
        ax.plot(T, results['SM'], 'k-', label='Equilibrium', linewidth=2)
    if 'SM_BCC_A2' in results:
        mask = ~np.isnan(results['SM_BCC_A2'])
        if np.any(mask):
            ax.plot(np.array(T)[mask], np.array(results['SM_BCC_A2'])[mask],
                    'b--', label='BCC_A2', linewidth=1.5)
    if 'SM_B2_221' in results:
        mask = ~np.isnan(results['SM_B2_221'])
        if np.any(mask):
            ax.plot(np.array(T)[mask], np.array(results['SM_B2_221'])[mask],
                    'r--', label='B2_221', linewidth=1.5)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Entropy (J/mol/K)')
    ax.set_title(f'Entropy vs Temperature\nx(V) = {x_v}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Heat capacity
    ax = axes[1, 1]
    if 'CPM' in results:
        ax.plot(T, results['CPM'], 'k-', label='Equilibrium', linewidth=2)
    if 'CPM_BCC_A2' in results:
        mask = ~np.isnan(results['CPM_BCC_A2'])
        if np.any(mask):
            ax.plot(np.array(T)[mask], np.array(results['CPM_BCC_A2'])[mask],
                    'b--', label='BCC_A2', linewidth=1.5)
    if 'CPM_B2_221' in results:
        mask = ~np.isnan(results['CPM_B2_221'])
        if np.any(mask):
            ax.plot(np.array(T)[mask], np.array(results['CPM_B2_221'])[mask],
                    'r--', label='B2_221', linewidth=1.5)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Heat Capacity (J/mol/K)')
    ax.set_title(f'Heat Capacity vs Temperature\nx(V) = {x_v}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(SCRIPT_DIR, f'{output_prefix}_xV{x_v:.2f}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved property diagram to: {output_path}")
    plt.close()


def plot_phase_fraction_vs_temperature(
    results: Dict[str, Any],
    x_v: float,
    output_file: str = 'Fe-V_phase_fraction_vs_T.png'
) -> None:
    """Plot phase fractions vs temperature."""
    if results is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    T = np.array(results['T'])
    np_bcc = np.array(results['NP_BCC_A2'])
    np_b2 = np.array(results['NP_B2_221'])

    ax.plot(T, np_bcc, 'b-', label='BCC_A2', linewidth=2)
    ax.plot(T, np_b2, 'r-', label='B2_221', linewidth=2)

    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Phase Fraction', fontsize=12)
    ax.set_title(f'Phase Fraction vs Temperature\nx(V) = {x_v}', fontsize=14)
    ax.legend(loc='best')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(SCRIPT_DIR, output_file)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved phase fraction diagram to: {output_path}")
    plt.close()


def plot_property_vs_composition(
    results: Dict[str, Any],
    temperature: float,
    output_prefix: str = 'Fe-V_property_vs_xV'
) -> None:
    """Plot thermodynamic properties vs composition."""
    if results is None:
        return

    x_V = np.array(results['x_V'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Gibbs energy
    ax = axes[0]
    if 'GM' in results:
        ax.plot(x_V, results['GM'], 'k-', label='Equilibrium', linewidth=2)
    if 'GM_BCC_A2' in results:
        mask = ~np.isnan(results['GM_BCC_A2'])
        if np.any(mask):
            ax.plot(np.array(x_V)[mask], np.array(results['GM_BCC_A2'])[mask],
                    'b--', label='BCC_A2', linewidth=1.5)
    if 'GM_B2_221' in results:
        mask = ~np.isnan(results['GM_B2_221'])
        if np.any(mask):
            ax.plot(np.array(x_V)[mask], np.array(results['GM_B2_221'])[mask],
                    'r--', label='B2_221', linewidth=1.5)
    ax.set_xlabel('Mole Fraction V')
    ax.set_ylabel('Gibbs Energy (J/mol)')
    ax.set_title(f'Gibbs Energy vs Composition\nT = {temperature} K')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Enthalpy
    ax = axes[1]
    if 'HM' in results:
        ax.plot(x_V, results['HM'], 'k-', label='Equilibrium', linewidth=2)
    if 'HM_BCC_A2' in results:
        mask = ~np.isnan(results['HM_BCC_A2'])
        if np.any(mask):
            ax.plot(np.array(x_V)[mask], np.array(results['HM_BCC_A2'])[mask],
                    'b--', label='BCC_A2', linewidth=1.5)
    if 'HM_B2_221' in results:
        mask = ~np.isnan(results['HM_B2_221'])
        if np.any(mask):
            ax.plot(np.array(x_V)[mask], np.array(results['HM_B2_221'])[mask],
                    'r--', label='B2_221', linewidth=1.5)
    ax.set_xlabel('Mole Fraction V')
    ax.set_ylabel('Enthalpy (J/mol)')
    ax.set_title(f'Enthalpy vs Composition\nT = {temperature} K')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[2]
    if 'SM' in results:
        ax.plot(x_V, results['SM'], 'k-', label='Equilibrium', linewidth=2)
    if 'SM_BCC_A2' in results:
        mask = ~np.isnan(results['SM_BCC_A2'])
        if np.any(mask):
            ax.plot(np.array(x_V)[mask], np.array(results['SM_BCC_A2'])[mask],
                    'b--', label='BCC_A2', linewidth=1.5)
    if 'SM_B2_221' in results:
        mask = ~np.isnan(results['SM_B2_221'])
        if np.any(mask):
            ax.plot(np.array(x_V)[mask], np.array(results['SM_B2_221'])[mask],
                    'r--', label='B2_221', linewidth=1.5)
    ax.set_xlabel('Mole Fraction V')
    ax.set_ylabel('Entropy (J/mol/K)')
    ax.set_title(f'Entropy vs Composition\nT = {temperature} K')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(SCRIPT_DIR, f'{output_prefix}_T{temperature:.0f}K.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved property diagram to: {output_path}")
    plt.close()


def main():
    """Main function."""
    print("=" * 60)
    print("Fe-V Property Diagram Calculation using TC-Python")
    print("=" * 60)

    if not TC_PYTHON_AVAILABLE:
        print("\nTC-Python is not installed. Exiting.")
        sys.exit(1)

    if not check_tdb_file():
        sys.exit(1)

    # Calculate properties vs temperature at x(V) = 0.5
    x_v = 0.5
    print(f"\n1. Calculating properties vs temperature at x(V) = {x_v}...")
    results_T = calculate_property_vs_temperature(x_v=x_v)
    if results_T:
        plot_property_vs_temperature(results_T, x_v)

    # Calculate phase fractions vs temperature
    print(f"\n2. Calculating phase fractions vs temperature at x(V) = {x_v}...")
    results_NP = calculate_phase_fraction_vs_temperature(x_v=x_v)
    if results_NP:
        plot_phase_fraction_vs_temperature(results_NP, x_v)

    # Calculate properties vs composition at T = 1000 K
    temperature = 1000
    print(f"\n3. Calculating properties vs composition at T = {temperature} K...")
    results_xV = calculate_property_vs_composition(temperature=temperature)
    if results_xV:
        plot_property_vs_composition(results_xV, temperature)

    print("\nDone!")


if __name__ == '__main__':
    main()
