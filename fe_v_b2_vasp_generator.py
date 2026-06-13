#!/usr/bin/env python3
"""
Fe-V B2-BCC VASP Input Generator

This script:
1. Fetches Fe-V B2-BCC structure data from Materials Project
2. Generates 2x2x1 BCC supercell (8 atoms)
3. Creates all 256 configurations (2^8 = 256 for Fe/V on each site)
4. Generates VASP input files (POSCAR, INCAR, KPOINTS)
5. Creates mpirun script for sequential calculations
6. Creates script to collect energies to CSV

Usage:
    python fe_v_b2_vasp_generator.py --api_key YOUR_API_KEY
"""

import os
import sys
import argparse
import itertools
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

# Materials Project and pymatgen imports
from mp_api.client import MPRester
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.vasp import Poscar, Incar, Kpoints


def fetch_fe_v_bcc_from_mp(api_key: str) -> Optional[Structure]:
    """
    Fetch Fe-V BCC structure from Materials Project.
    
    Args:
        api_key: Materials Project API key
        
    Returns:
        BCC structure or None if not found
    """
    print("Fetching Fe-V B2-BCC data from Materials Project...")
    
    with MPRester(api_key) as mpr:
        # Search for Fe-V compounds with BCC structure
        # First try to get pure Fe BCC as base structure
        try:
            docs = mpr.materials.summary.search(
                chemsys="Fe",
                spacegroup_symbol="Im-3m",  # BCC space group
                fields=["material_id", "structure", "formula_pretty", "energy_per_atom"]
            )
            
            if docs:
                print(f"Found {len(docs)} Fe BCC structures")
                # Get the most stable one (lowest energy)
                best_doc = min(docs, key=lambda x: x.energy_per_atom if x.energy_per_atom else float('inf'))
                print(f"Using {best_doc.material_id}: {best_doc.formula_pretty}")
                return best_doc.structure
            
        except Exception as e:
            print(f"Error fetching from Materials Project: {e}")
            print("Using default BCC structure...")
    
    return None


def create_bcc_unit_cell(lattice_constant: float = 2.87) -> Structure:
    """
    Create a BCC unit cell.
    
    Args:
        lattice_constant: Lattice constant in Angstrom (default: 2.87 for Fe)
        
    Returns:
        BCC unit cell structure
    """
    lattice = Lattice.cubic(lattice_constant)
    # BCC has 2 atoms: (0,0,0) and (0.5,0.5,0.5)
    species = ["Fe", "Fe"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    
    return Structure(lattice, species, coords)


def create_2x2x1_supercell(base_structure: Structure) -> Structure:
    """
    Create a 2x2x1 supercell from the base BCC structure.
    
    Args:
        base_structure: Base BCC unit cell
        
    Returns:
        2x2x1 supercell with 8 atoms
    """
    # Create 2x2x1 supercell
    supercell = base_structure.copy()
    supercell.make_supercell([2, 2, 1])
    
    print(f"Created 2x2x1 supercell with {len(supercell)} atoms")
    return supercell


def generate_all_configurations(supercell: Structure) -> List[Tuple[Structure, str, int]]:
    """
    Generate all 256 configurations by substituting Fe with V.
    
    Args:
        supercell: Base supercell structure
        
    Returns:
        List of (structure, configuration_name, n_vanadium) tuples
    """
    n_atoms = len(supercell)
    configurations = []
    
    # Generate all 2^8 = 256 configurations
    for i in range(2**n_atoms):
        # Convert index to binary representation
        binary = format(i, f'0{n_atoms}b')
        
        # Create new structure
        new_structure = supercell.copy()
        
        # Replace atoms based on binary representation
        # 0 = Fe, 1 = V
        n_v = 0
        for j, bit in enumerate(binary):
            if bit == '1':
                new_structure.replace(j, Element("V"))
                n_v += 1
        
        # Create configuration name
        config_name = f"config_{i:03d}_Fe{n_atoms-n_v}V{n_v}"
        configurations.append((new_structure, config_name, n_v))
    
    print(f"Generated {len(configurations)} configurations")
    return configurations


def create_incar(is_relaxation: bool = True) -> Incar:
    """
    Create INCAR file for VASP calculation.
    
    Args:
        is_relaxation: Whether this is a relaxation calculation
        
    Returns:
        Incar object
    """
    incar_params = {
        # General settings
        "SYSTEM": "Fe-V B2-BCC",
        "PREC": "Accurate",
        "ENCUT": 400,  # Energy cutoff in eV
        "EDIFF": 1E-6,  # Electronic convergence
        "EDIFFG": -0.01,  # Ionic convergence (force criterion)
        
        # Electronic settings
        "ISMEAR": 1,  # Methfessel-Paxton smearing
        "SIGMA": 0.1,  # Smearing width
        "LREAL": "Auto",  # Real space projection
        
        # Ionic relaxation
        "IBRION": 2 if is_relaxation else -1,  # CG relaxation or static
        "NSW": 100 if is_relaxation else 0,  # Number of ionic steps
        "ISIF": 3 if is_relaxation else 2,  # Relax cell shape and volume
        
        # Spin polarization (important for Fe)
        "ISPIN": 2,  # Spin-polarized calculation
        "MAGMOM": "8*2.0",  # Initial magnetic moments
        
        # Output
        "LWAVE": False,  # Don't write WAVECAR
        "LCHARG": False,  # Don't write CHGCAR
        "LORBIT": 11,  # Write DOSCAR and PROCAR
        
        # Parallelization
        "NCORE": 4,  # Cores per band
    }
    
    return Incar(incar_params)


def create_kpoints(mesh: Tuple[int, int, int] = (6, 6, 8)) -> Kpoints:
    """
    Create KPOINTS file for VASP calculation.
    
    Args:
        mesh: K-point mesh (default: 6x6x8 for 2x2x1 supercell)
        
    Returns:
        Kpoints object
    """
    return Kpoints.gamma_automatic(mesh)


def write_vasp_inputs(
    structure: Structure,
    config_name: str,
    output_dir: Path,
    incar: Incar,
    kpoints: Kpoints
) -> None:
    """
    Write VASP input files for a configuration.
    
    Args:
        structure: Structure to write
        config_name: Configuration name
        output_dir: Output directory
        incar: INCAR parameters
        kpoints: KPOINTS parameters
    """
    config_dir = output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Write POSCAR
    poscar = Poscar(structure)
    poscar.write_file(config_dir / "POSCAR")
    
    # Write INCAR
    incar.write_file(config_dir / "INCAR")
    
    # Write KPOINTS
    kpoints.write_file(config_dir / "KPOINTS")
    
    # Write POTCAR info (user needs to generate actual POTCAR)
    potcar_info = config_dir / "POTCAR_INFO.txt"
    with open(potcar_info, 'w') as f:
        f.write("# POTCAR Information\n")
        f.write("# You need to generate POTCAR using your VASP pseudopotential library\n")
        f.write("# Required pseudopotentials:\n")
        elements = list(set([str(site.specie) for site in structure]))
        elements.sort()
        for elem in elements:
            f.write(f"#   {elem}_pv (recommended) or {elem}\n")
        f.write("\n# Example command:\n")
        f.write("# cat ")
        for elem in elements:
            f.write(f"$VASP_PP_PATH/PBE/{elem}_pv/POTCAR ")
        f.write("> POTCAR\n")


def create_mpirun_script(output_dir: Path, n_procs: int = 16) -> None:
    """
    Create mpirun script for sequential VASP calculations.
    
    Args:
        output_dir: Output directory containing all configurations
        n_procs: Number of MPI processes
    """
    script_path = output_dir / "run_all_vasp.sh"
    
    script_content = f'''#!/bin/bash
#
# Fe-V B2-BCC VASP Calculation Script
# This script runs VASP calculations for all 256 configurations sequentially
#
# Usage: ./run_all_vasp.sh
#
# Prerequisites:
# 1. VASP executable in PATH or set VASP_CMD variable
# 2. POTCAR files generated in each directory
# 3. Sufficient computational resources
#

# Configuration
VASP_CMD="${{VASP_CMD:-vasp_std}}"
NPROCS={n_procs}
LOG_FILE="calculation_log.txt"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"

# Initialize log file
echo "Fe-V B2-BCC VASP Calculations" > "$LOG_FILE"
echo "Started at: $(date)" >> "$LOG_FILE"
echo "Number of processes: $NPROCS" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Counter for progress
TOTAL=256
CURRENT=0

# Loop through all configuration directories
for CONFIG_DIR in config_*/; do
    if [ -d "$CONFIG_DIR" ]; then
        CURRENT=$((CURRENT + 1))
        CONFIG_NAME=$(basename "$CONFIG_DIR")
        
        echo ""
        echo "=========================================="
        echo "[$CURRENT/$TOTAL] Processing: $CONFIG_NAME"
        echo "=========================================="
        
        cd "$CONFIG_DIR"
        
        # Check if POTCAR exists
        if [ ! -f "POTCAR" ]; then
            echo "WARNING: POTCAR not found in $CONFIG_NAME. Skipping..."
            echo "[$CONFIG_NAME] SKIPPED - No POTCAR" >> "../$LOG_FILE"
            cd ..
            continue
        fi
        
        # Check if calculation already completed
        if [ -f "OUTCAR" ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
            echo "Calculation already completed. Skipping..."
            echo "[$CONFIG_NAME] SKIPPED - Already completed" >> "../$LOG_FILE"
            cd ..
            continue
        fi
        
        # Run VASP
        START_TIME=$(date +%s)
        echo "Starting VASP calculation at $(date)"
        
        mpirun -np $NPROCS $VASP_CMD > vasp.out 2>&1
        VASP_EXIT_CODE=$?
        
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        
        # Check if calculation succeeded
        if [ $VASP_EXIT_CODE -eq 0 ] && [ -f "OUTCAR" ]; then
            if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
                # Extract total energy
                ENERGY=$(grep "free  energy   TOTEN" OUTCAR | tail -1 | awk '{{print $5}}')
                echo "Calculation completed successfully!"
                echo "Total energy: $ENERGY eV"
                echo "Elapsed time: $ELAPSED seconds"
                echo "[$CONFIG_NAME] SUCCESS - E=$ENERGY eV - Time=${{ELAPSED}}s" >> "../$LOG_FILE"
            else
                echo "WARNING: Calculation may not have converged"
                echo "[$CONFIG_NAME] WARNING - May not have converged" >> "../$LOG_FILE"
            fi
        else
            echo "ERROR: VASP calculation failed!"
            echo "[$CONFIG_NAME] FAILED - Exit code: $VASP_EXIT_CODE" >> "../$LOG_FILE"
        fi
        
        cd ..
    fi
done

echo ""
echo "=========================================="
echo "All calculations completed at $(date)"
echo "=========================================="
echo "" >> "$LOG_FILE"
echo "Completed at: $(date)" >> "$LOG_FILE"

# Run energy collection script
if [ -f "collect_energies.py" ]; then
    echo "Collecting energies to CSV..."
    python3 collect_energies.py
fi

echo "Done!"
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    print(f"Created mpirun script: {script_path}")


def create_energy_collection_script(output_dir: Path) -> None:
    """
    Create Python script to collect energies from VASP calculations.
    
    Args:
        output_dir: Output directory containing all configurations
    """
    script_path = output_dir / "collect_energies.py"
    
    script_content = '''#!/usr/bin/env python3
"""
Collect total energies from VASP calculations and save to CSV.

Usage:
    python collect_energies.py
"""

import os
import re
import csv
from pathlib import Path
from datetime import datetime


def parse_outcar(outcar_path: Path) -> dict:
    """
    Parse OUTCAR file to extract calculation results.
    
    Args:
        outcar_path: Path to OUTCAR file
        
    Returns:
        Dictionary with calculation results
    """
    results = {
        'converged': False,
        'total_energy': None,
        'energy_per_atom': None,
        'n_atoms': None,
        'n_iterations': None,
        'total_magnetization': None,
    }
    
    if not outcar_path.exists():
        return results
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Check convergence
        if 'reached required accuracy' in content:
            results['converged'] = True
        
        # Extract total energy (last occurrence)
        energy_matches = re.findall(r'free  energy   TOTEN\\s*=\\s*([\\d.-]+)', content)
        if energy_matches:
            results['total_energy'] = float(energy_matches[-1])
        
        # Extract number of atoms
        n_atoms_match = re.search(r'NIONS\\s*=\\s*(\\d+)', content)
        if n_atoms_match:
            results['n_atoms'] = int(n_atoms_match.group(1))
            if results['total_energy'] is not None:
                results['energy_per_atom'] = results['total_energy'] / results['n_atoms']
        
        # Extract number of iterations
        n_iter_matches = re.findall(r'Iteration\\s+(\\d+)', content)
        if n_iter_matches:
            results['n_iterations'] = int(n_iter_matches[-1])
        
        # Extract total magnetization
        mag_matches = re.findall(r'number of electron\\s+[\\d.]+\\s+magnetization\\s+([\\d.-]+)', content)
        if mag_matches:
            results['total_magnetization'] = float(mag_matches[-1])
            
    except Exception as e:
        print(f"Error parsing {outcar_path}: {e}")
    
    return results


def parse_poscar(poscar_path: Path) -> dict:
    """
    Parse POSCAR file to extract composition.
    
    Args:
        poscar_path: Path to POSCAR file
        
    Returns:
        Dictionary with composition info
    """
    results = {
        'elements': [],
        'n_fe': 0,
        'n_v': 0,
        'composition': '',
    }
    
    if not poscar_path.exists():
        return results
    
    try:
        with open(poscar_path, 'r') as f:
            lines = f.readlines()
        
        # Element names are on line 6 (0-indexed: 5)
        # Atom counts are on line 7 (0-indexed: 6)
        if len(lines) >= 7:
            elements = lines[5].split()
            counts = [int(x) for x in lines[6].split()]
            
            results['elements'] = elements
            for elem, count in zip(elements, counts):
                if elem == 'Fe':
                    results['n_fe'] = count
                elif elem == 'V':
                    results['n_v'] = count
            
            results['composition'] = f"Fe{results['n_fe']}V{results['n_v']}"
            
    except Exception as e:
        print(f"Error parsing {poscar_path}: {e}")
    
    return results


def collect_all_energies(base_dir: Path = None) -> list:
    """
    Collect energies from all configuration directories.
    
    Args:
        base_dir: Base directory containing configuration folders
        
    Returns:
        List of dictionaries with results
    """
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    results = []
    
    # Find all configuration directories
    config_dirs = sorted(base_dir.glob('config_*'))
    
    print(f"Found {len(config_dirs)} configuration directories")
    
    for config_dir in config_dirs:
        if not config_dir.is_dir():
            continue
        
        config_name = config_dir.name
        
        # Parse POSCAR for composition
        poscar_info = parse_poscar(config_dir / 'POSCAR')
        
        # Parse OUTCAR for energy
        outcar_info = parse_outcar(config_dir / 'OUTCAR')
        
        # Combine results
        result = {
            'config_name': config_name,
            'config_index': int(config_name.split('_')[1]),
            **poscar_info,
            **outcar_info,
        }
        
        results.append(result)
    
    return results


def save_to_csv(results: list, output_path: Path) -> None:
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Output CSV file path
    """
    if not results:
        print("No results to save")
        return
    
    # Define column order
    columns = [
        'config_index',
        'config_name',
        'composition',
        'n_fe',
        'n_v',
        'converged',
        'total_energy',
        'energy_per_atom',
        'n_atoms',
        'n_iterations',
        'total_magnetization',
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        
        # Sort by config_index
        sorted_results = sorted(results, key=lambda x: x.get('config_index', 0))
        writer.writerows(sorted_results)
    
    print(f"Saved results to {output_path}")


def print_summary(results: list) -> None:
    """
    Print summary of calculation results.
    
    Args:
        results: List of result dictionaries
    """
    total = len(results)
    converged = sum(1 for r in results if r.get('converged', False))
    has_energy = sum(1 for r in results if r.get('total_energy') is not None)
    
    print("\\n" + "=" * 50)
    print("CALCULATION SUMMARY")
    print("=" * 50)
    print(f"Total configurations: {total}")
    print(f"Converged: {converged}")
    print(f"With energy data: {has_energy}")
    
    if has_energy > 0:
        energies = [r['total_energy'] for r in results if r.get('total_energy') is not None]
        print(f"\\nEnergy range: {min(energies):.6f} to {max(energies):.6f} eV")
        
        # Group by composition
        print("\\nEnergies by composition:")
        compositions = {}
        for r in results:
            if r.get('total_energy') is not None:
                comp = r.get('composition', 'Unknown')
                if comp not in compositions:
                    compositions[comp] = []
                compositions[comp].append(r['total_energy'])
        
        for comp in sorted(compositions.keys()):
            energies = compositions[comp]
            print(f"  {comp}: n={len(energies)}, min={min(energies):.6f}, max={max(energies):.6f} eV")


def main():
    """Main function."""
    print("Fe-V B2-BCC Energy Collection Script")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get base directory
    base_dir = Path(__file__).parent
    
    # Collect all energies
    results = collect_all_energies(base_dir)
    
    # Save to CSV
    output_csv = base_dir / 'fe_v_b2_energies.csv'
    save_to_csv(results, output_csv)
    
    # Print summary
    print_summary(results)
    
    print("\\nDone!")


if __name__ == '__main__':
    main()
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    print(f"Created energy collection script: {script_path}")


def create_zip_archive(output_dir: Path, zip_path: Path) -> None:
    """
    Create ZIP archive of all VASP input files.
    
    Args:
        output_dir: Directory containing VASP inputs
        zip_path: Output ZIP file path
    """
    print(f"Creating ZIP archive: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(output_dir.parent)
                zipf.write(file_path, arcname)
    
    # Get file size
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Created ZIP archive: {zip_path} ({size_mb:.2f} MB)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate VASP input files for Fe-V B2-BCC 2x2x1 supercell configurations'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        required=True,
        help='Materials Project API key'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='fe_v_b2_vasp_inputs',
        help='Output directory name (default: fe_v_b2_vasp_inputs)'
    )
    parser.add_argument(
        '--lattice_constant',
        type=float,
        default=2.87,
        help='BCC lattice constant in Angstrom (default: 2.87 for Fe)'
    )
    parser.add_argument(
        '--n_procs',
        type=int,
        default=16,
        help='Number of MPI processes for VASP (default: 16)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Fe-V B2-BCC VASP Input Generator")
    print("=" * 60)
    print()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Step 1: Fetch or create BCC structure
    print("\n--- Step 1: Fetching/Creating BCC Structure ---")
    mp_structure = fetch_fe_v_bcc_from_mp(args.api_key)
    
    if mp_structure is not None:
        # Use Materials Project structure as reference
        # Extract lattice constant
        lattice_constant = mp_structure.lattice.a
        print(f"Using lattice constant from Materials Project: {lattice_constant:.4f} A")
    else:
        lattice_constant = args.lattice_constant
        print(f"Using default lattice constant: {lattice_constant:.4f} A")
    
    # Create BCC unit cell
    bcc_unit = create_bcc_unit_cell(lattice_constant)
    print(f"Created BCC unit cell with lattice constant: {lattice_constant:.4f} A")
    
    # Step 2: Create 2x2x1 supercell
    print("\n--- Step 2: Creating 2x2x1 Supercell ---")
    supercell = create_2x2x1_supercell(bcc_unit)
    
    # Step 3: Generate all 256 configurations
    print("\n--- Step 3: Generating All Configurations ---")
    configurations = generate_all_configurations(supercell)
    
    # Step 4: Create VASP input files
    print("\n--- Step 4: Creating VASP Input Files ---")
    incar = create_incar(is_relaxation=True)
    kpoints = create_kpoints()
    
    for structure, config_name, n_v in configurations:
        write_vasp_inputs(structure, config_name, output_dir, incar, kpoints)
    
    print(f"Created VASP inputs for {len(configurations)} configurations")
    
    # Step 5: Create mpirun script
    print("\n--- Step 5: Creating mpirun Script ---")
    create_mpirun_script(output_dir, args.n_procs)
    
    # Step 6: Create energy collection script
    print("\n--- Step 6: Creating Energy Collection Script ---")
    create_energy_collection_script(output_dir)
    
    # Step 7: Create ZIP archive
    print("\n--- Step 7: Creating ZIP Archive ---")
    zip_path = Path(f"{args.output_dir}.zip")
    create_zip_archive(output_dir, zip_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total configurations: {len(configurations)}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"ZIP archive: {zip_path.absolute()}")
    print()
    print("Configuration breakdown by V content:")
    v_counts = {}
    for _, _, n_v in configurations:
        v_counts[n_v] = v_counts.get(n_v, 0) + 1
    for n_v in sorted(v_counts.keys()):
        n_fe = 8 - n_v
        print(f"  Fe{n_fe}V{n_v}: {v_counts[n_v]} configurations")
    
    print()
    print("Next steps:")
    print("1. Generate POTCAR files in each configuration directory")
    print("2. Run: ./run_all_vasp.sh")
    print("3. After calculations complete, energies will be in fe_v_b2_energies.csv")
    print()
    print("Done!")


if __name__ == '__main__':
    main()
