#!/usr/bin/env python3
"""
Fe-V BCC Phase Diagram Calculation using pycalphad

This script calculates the metastable BCC phase diagram for the Fe-V system,
considering only BCC_A2 and B2_221 phases (excluding FCC_A1 and SIGMA).

Usage:
    python fev_bcc_pycalphad.py

Output:
    - Fe-V_BCC_phase_diagram.png: Phase diagram plot
    - Fe-V_BCC_phase_data.csv: Phase boundary data
"""

import numpy as np
import matplotlib.pyplot as plt
from pycalphad import Database, equilibrium, variables as v
import pandas as pd
import os

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TDB_FILE = os.path.join(SCRIPT_DIR, 'Fe-V_B2_221.tdb')

def load_database():
    """Load the Fe-V TDB database."""
    print(f"Loading database from: {TDB_FILE}")
    db = Database(TDB_FILE)
    print(f"Available phases: {list(db.phases.keys())}")
    return db

def calculate_bcc_phase_diagram(db, T_min=800, T_max=2000, T_step=25, 
                                 x_V_min=0.0, x_V_max=1.0, x_V_step=0.02):
    """
    Calculate the metastable BCC phase diagram.
    
    Parameters:
    -----------
    db : Database
        pycalphad Database object
    T_min, T_max : float
        Temperature range in K
    T_step : float
        Temperature step in K
    x_V_min, x_V_max : float
        V composition range (mole fraction)
    x_V_step : float
        Composition step
    
    Returns:
    --------
    results : dict
        Dictionary containing phase diagram data
    """
    # Define components and phases
    components = ['FE', 'V', 'VA']
    
    # BCC phases only (metastable diagram)
    bcc_phases = ['BCC_A2', 'B2_221']
    
    # Check which phases are available in the database
    available_phases = [p for p in bcc_phases if p in db.phases]
    print(f"Using phases: {available_phases}")
    
    if not available_phases:
        print("Warning: No BCC phases found in database!")
        return None
    
    # Temperature and composition grids
    temperatures = np.arange(T_min, T_max + T_step, T_step)
    compositions = np.arange(x_V_min, x_V_max + x_V_step, x_V_step)
    
    print(f"Calculating equilibrium for {len(temperatures)} temperatures "
          f"and {len(compositions)} compositions...")
    
    # Store results
    results = {
        'T': [],
        'x_V': [],
        'phases': [],
        'phase_fractions': []
    }
    
    # Calculate equilibrium at each point
    total_points = len(temperatures) * len(compositions)
    point_count = 0
    
    for T in temperatures:
        for x_V in compositions:
            point_count += 1
            if point_count % 100 == 0:
                print(f"  Progress: {point_count}/{total_points} "
                      f"({100*point_count/total_points:.1f}%)")
            
            try:
                # Set conditions
                conditions = {
                    v.T: T,
                    v.P: 101325,  # 1 atm
                    v.X('V'): x_V if x_V > 0 else 1e-10,
                    v.N: 1
                }
                
                # Calculate equilibrium
                eq = equilibrium(db, components, available_phases, conditions)
                
                # Extract phase information
                phase_names = []
                phase_fracs = []
                
                # Get phase fractions from the equilibrium result
                for phase in available_phases:
                    try:
                        nf = float(eq.NP.sel(phase=phase).values.flatten()[0])
                        if nf > 0.01:  # Only include phases with >1% fraction
                            phase_names.append(phase)
                            phase_fracs.append(nf)
                    except:
                        pass
                
                results['T'].append(T)
                results['x_V'].append(x_V)
                results['phases'].append('+'.join(phase_names) if phase_names else 'unknown')
                results['phase_fractions'].append(phase_fracs)
                
            except Exception as e:
                results['T'].append(T)
                results['x_V'].append(x_V)
                results['phases'].append('error')
                results['phase_fractions'].append([])
    
    return results

def plot_phase_diagram(results, output_file='Fe-V_BCC_phase_diagram.png'):
    """
    Plot the BCC phase diagram.
    
    Parameters:
    -----------
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
    ax.set_title('Fe-V Metastable BCC Phase Diagram\n(BCC_A2 + B2_221 only)', fontsize=14)
    ax.legend(loc='best')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(SCRIPT_DIR, output_file)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved phase diagram to: {output_path}")
    plt.close()

def save_phase_data(results, output_file='Fe-V_BCC_phase_data.csv'):
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

def main():
    """Main function."""
    print("=" * 60)
    print("Fe-V BCC Phase Diagram Calculation")
    print("=" * 60)
    
    # Load database
    db = load_database()
    
    # Calculate phase diagram
    print("\nCalculating metastable BCC phase diagram...")
    results = calculate_bcc_phase_diagram(
        db,
        T_min=800,
        T_max=2000,
        T_step=50,
        x_V_min=0.0,
        x_V_max=1.0,
        x_V_step=0.05
    )
    
    # Plot and save results
    print("\nGenerating plots...")
    plot_phase_diagram(results)
    save_phase_data(results)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
