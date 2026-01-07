#!/usr/bin/env python3
"""
Extract lattice constants from VASP output files (CONTCAR/POSCAR).

This script reads VASP structure files and extracts lattice parameters
for Fe-V B2_221 supercell calculations.

Usage:
    python extract_lattice_constants.py <directory_with_calculations>
    
    or import as module:
    from extract_lattice_constants import extract_lattice_from_contcar

Output:
    CSV file with config_index, composition, lattice parameters (a, b, c), 
    volume, and volume per atom.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def read_poscar_lattice(filepath):
    """
    Read lattice vectors from POSCAR/CONTCAR file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to POSCAR or CONTCAR file
        
    Returns:
    --------
    dict with keys:
        'lattice_vectors': 3x3 numpy array of lattice vectors
        'a', 'b', 'c': lattice parameters (Angstrom)
        'alpha', 'beta', 'gamma': lattice angles (degrees)
        'volume': cell volume (Angstrom^3)
        'scale_factor': VASP scale factor
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Line 1: comment
    comment = lines[0].strip()
    
    # Line 2: scale factor
    scale_factor = float(lines[1].strip())
    
    # Lines 3-5: lattice vectors
    lattice_vectors = np.zeros((3, 3))
    for i in range(3):
        parts = lines[2 + i].split()
        lattice_vectors[i] = [float(x) for x in parts[:3]]
    
    # Apply scale factor
    lattice_vectors *= scale_factor
    
    # Calculate lattice parameters
    a_vec = lattice_vectors[0]
    b_vec = lattice_vectors[1]
    c_vec = lattice_vectors[2]
    
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)
    
    # Calculate angles
    alpha = np.degrees(np.arccos(np.dot(b_vec, c_vec) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(a_vec, c_vec) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(a_vec, b_vec) / (a * b)))
    
    # Calculate volume
    volume = np.abs(np.dot(a_vec, np.cross(b_vec, c_vec)))
    
    return {
        'lattice_vectors': lattice_vectors,
        'a': a,
        'b': b,
        'c': c,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'volume': volume,
        'scale_factor': scale_factor,
        'comment': comment
    }


def extract_lattice_from_contcar(contcar_path):
    """
    Extract lattice constant from CONTCAR file.
    
    For B2_221 (2x2x1 supercell of BCC), the effective BCC lattice constant
    is calculated from the supercell dimensions.
    
    Parameters:
    -----------
    contcar_path : str or Path
        Path to CONTCAR file
        
    Returns:
    --------
    dict with lattice information including effective BCC lattice constant
    """
    result = read_poscar_lattice(contcar_path)
    
    # For 2x2x1 BCC supercell:
    # a_supercell = 2 * a_BCC
    # b_supercell = 2 * a_BCC  
    # c_supercell = 1 * a_BCC
    # So effective a_BCC = (a/2 + b/2 + c) / 3 for cubic approximation
    # Or more accurately: a_BCC = (volume / 8)^(1/3) * 2^(1/3) for BCC with 8 atoms
    
    # For 8-atom BCC supercell: V = 2 * a_BCC^3 (BCC has 2 atoms per unit cell)
    # So a_BCC = (V / 2)^(1/3)
    # But for 2x2x1 supercell with 8 atoms: V_supercell = 4 * V_BCC = 4 * 2 * a_BCC^3 = 8 * a_BCC^3
    # Wait, let me recalculate:
    # BCC unit cell: 2 atoms, volume = a^3
    # 2x2x1 supercell: 2*2*1 = 4 unit cells, 8 atoms, volume = 4 * a^3
    # So a_BCC = (V_supercell / 4)^(1/3)
    
    volume = result['volume']
    n_atoms = 8  # B2_221 has 8 atoms
    
    # Effective BCC lattice constant
    # V_supercell = 4 * a_BCC^3 for 2x2x1 supercell
    a_bcc = (volume / 4) ** (1/3)
    
    # Volume per atom
    volume_per_atom = volume / n_atoms
    
    result['a_bcc_effective'] = a_bcc
    result['volume_per_atom'] = volume_per_atom
    result['n_atoms'] = n_atoms
    
    return result


def process_calculation_directory(calc_dir, config_index=None, config_name=None):
    """
    Process a single calculation directory and extract lattice information.
    
    Parameters:
    -----------
    calc_dir : str or Path
        Path to calculation directory containing CONTCAR or POSCAR
    config_index : int, optional
        Configuration index
    config_name : str, optional
        Configuration name
        
    Returns:
    --------
    dict with all extracted information, or None if no structure file found
    """
    calc_dir = Path(calc_dir)
    
    # Try CONTCAR first (relaxed structure), then POSCAR (initial structure)
    contcar_path = calc_dir / 'CONTCAR'
    poscar_path = calc_dir / 'POSCAR'
    
    if contcar_path.exists() and contcar_path.stat().st_size > 0:
        structure_file = contcar_path
        structure_type = 'CONTCAR'
    elif poscar_path.exists():
        structure_file = poscar_path
        structure_type = 'POSCAR'
    else:
        return None
    
    try:
        lattice_info = extract_lattice_from_contcar(structure_file)
    except Exception as e:
        print(f"Error reading {structure_file}: {e}")
        return None
    
    result = {
        'config_index': config_index,
        'config_name': config_name,
        'structure_file': structure_type,
        'a': lattice_info['a'],
        'b': lattice_info['b'],
        'c': lattice_info['c'],
        'alpha': lattice_info['alpha'],
        'beta': lattice_info['beta'],
        'gamma': lattice_info['gamma'],
        'volume': lattice_info['volume'],
        'volume_per_atom': lattice_info['volume_per_atom'],
        'a_bcc_effective': lattice_info['a_bcc_effective'],
    }
    
    return result


def scan_calculations_directory(base_dir, pattern='config_*'):
    """
    Scan a directory for calculation subdirectories and extract lattice constants.
    
    Parameters:
    -----------
    base_dir : str or Path
        Base directory containing calculation subdirectories
    pattern : str
        Glob pattern for calculation directories (default: 'config_*')
        
    Returns:
    --------
    pandas DataFrame with lattice information for all configurations
    """
    base_dir = Path(base_dir)
    results = []
    
    # Find all calculation directories
    calc_dirs = sorted(base_dir.glob(pattern))
    
    for calc_dir in calc_dirs:
        if not calc_dir.is_dir():
            continue
        
        # Try to extract config_index from directory name
        dir_name = calc_dir.name
        config_index = None
        config_name = dir_name
        
        # Try to parse config_XXX_FeYVZ format
        if dir_name.startswith('config_'):
            parts = dir_name.split('_')
            if len(parts) >= 2:
                try:
                    config_index = int(parts[1])
                except ValueError:
                    pass
        
        result = process_calculation_directory(calc_dir, config_index, config_name)
        if result:
            results.append(result)
    
    if not results:
        print(f"No calculation directories found in {base_dir}")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    return df


def add_composition_info(df, n_atoms=8):
    """
    Add composition information based on config_index (bitmask encoding).
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with config_index column
    n_atoms : int
        Number of atoms in supercell (default: 8)
        
    Returns:
    --------
    DataFrame with added n_fe, n_v, x_v columns
    """
    if 'config_index' not in df.columns:
        return df
    
    # config_index is a bitmask where bit=1 means V, bit=0 means Fe
    df['n_v'] = df['config_index'].apply(lambda x: bin(int(x)).count('1') if pd.notna(x) else None)
    df['n_fe'] = n_atoms - df['n_v']
    df['x_v'] = df['n_v'] / n_atoms
    df['composition'] = df.apply(lambda row: f"Fe{int(row['n_fe'])}V{int(row['n_v'])}" if pd.notna(row['n_v']) else None, axis=1)
    
    return df


def calculate_vegard_lattice(df, a_fe=2.822, a_v=2.992):
    """
    Calculate Vegard's law prediction for lattice constant.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with x_v column
    a_fe : float
        BCC lattice constant for pure Fe (default: 2.822 Angstrom, Wang et al.)
    a_v : float
        BCC lattice constant for pure V (default: 2.992 Angstrom, Wang et al.)
        
    Returns:
    --------
    DataFrame with added a_vegard and a_deviation columns
    """
    if 'x_v' not in df.columns:
        return df
    
    # Vegard's law: a(x) = (1-x)*a_Fe + x*a_V
    df['a_vegard'] = (1 - df['x_v']) * a_fe + df['x_v'] * a_v
    df['a_deviation'] = df['a_bcc_effective'] - df['a_vegard']
    df['a_deviation_percent'] = (df['a_deviation'] / df['a_vegard']) * 100
    
    return df


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample usage:")
        print("  python extract_lattice_constants.py /path/to/calculations/")
        print("  python extract_lattice_constants.py /path/to/calculations/ --output lattice_data.csv")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    output_file = 'lattice_constants.csv'
    
    if len(sys.argv) >= 4 and sys.argv[2] == '--output':
        output_file = sys.argv[3]
    
    print(f"Scanning directory: {base_dir}")
    df = scan_calculations_directory(base_dir)
    
    if df.empty:
        print("No data extracted.")
        sys.exit(1)
    
    # Add composition info
    df = add_composition_info(df)
    
    # Calculate Vegard's law prediction
    df = calculate_vegard_lattice(df)
    
    # Sort by config_index
    if 'config_index' in df.columns:
        df = df.sort_values('config_index')
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nSaved lattice data to: {output_file}")
    print(f"Total configurations processed: {len(df)}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Columns: {list(df.columns)}")
    if 'a_bcc_effective' in df.columns:
        print(f"a_BCC range: {df['a_bcc_effective'].min():.4f} - {df['a_bcc_effective'].max():.4f} Angstrom")
    if 'volume_per_atom' in df.columns:
        print(f"Volume/atom range: {df['volume_per_atom'].min():.4f} - {df['volume_per_atom'].max():.4f} Angstrom^3")


if __name__ == '__main__':
    main()
