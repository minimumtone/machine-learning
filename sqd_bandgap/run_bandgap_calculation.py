#!/usr/bin/env python3
"""
SQD Band Gap Calculation Script

This script demonstrates the SQD method for computing band gaps of periodic materials.
It uses pre-computed DFT+U+V data and quantum hardware/simulator results to calculate
the band gap of HfO2 (hafnium dioxide).

Usage:
    python run_bandgap_calculation.py --material hafnium_2 --results_path /path/to/results

Reference:
    "Computing band gaps of periodic materials via sample-based quantum diagonalization"
    arXiv:2503.10901
"""

import argparse
import json
import os
import sys

from bandgap import (
    calculate_bandgap,
    compare_with_experiment,
    EXPERIMENTAL_BANDGAPS,
    DFT_UV_BANDGAPS
)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate band gap using SQD method"
    )
    parser.add_argument(
        '--material', 
        type=str, 
        default='hafnium_2',
        help='Material name (default: hafnium_2 for HfO2)'
    )
    parser.add_argument(
        '--results_path',
        type=str,
        default=None,
        help='Path to SQD results folder'
    )
    parser.add_argument(
        '--ne_neutral',
        type=int,
        default=24,
        help='Number of electrons in neutral system (default: 24 for HfO2)'
    )
    parser.add_argument(
        '--sampling',
        type=str,
        default='hardware',
        help='Sampling method: hardware, ffsim (default: hardware)'
    )
    
    args = parser.parse_args()
    
    # Map material names to experimental values
    material_map = {
        'hafnium_2': 'HfO2',
        'zirconia_2': 'ZrO2',
    }
    
    material_name = material_map.get(args.material, args.material)
    
    print("=" * 60)
    print(f"SQD Band Gap Calculation for {material_name}")
    print("=" * 60)
    
    if args.results_path:
        # Load results from specified path
        results_folder = os.path.join(args.results_path, args.material)
        
        energies = {}
        for ne in [args.ne_neutral - 1, args.ne_neutral, args.ne_neutral + 1]:
            ne_folder = os.path.join(
                results_folder, f"{ne}e", 
                f"results_sqd_{args.sampling}", "dicts"
            )
            
            if not os.path.exists(ne_folder):
                print(f"Warning: Results folder not found: {ne_folder}")
                continue
            
            files = sorted([f for f in os.listdir(ne_folder) 
                          if f.startswith("results_dict_")])
            if files:
                latest_file = files[-1]
                with open(os.path.join(ne_folder, latest_file)) as f:
                    data = json.load(f)
                energies[ne] = data['sqd_energy']
                print(f"Loaded {ne}e: E = {energies[ne]:.6f} eV")
        
        if len(energies) == 3:
            ne = args.ne_neutral
            bandgap = calculate_bandgap(
                energies[ne - 1], 
                energies[ne], 
                energies[ne + 1]
            )
        else:
            print("Error: Could not load all required energies")
            sys.exit(1)
    else:
        # Use pre-computed results from the paper
        print("\nUsing pre-computed results from the paper...")
        
        if args.material == 'hafnium_2':
            # HfO2 results from quantum hardware
            energies = {
                23: -389.5372901561972,  # Ne - 1
                24: -389.4330963086967,  # Ne
                25: -383.66675745876273  # Ne + 1
            }
        elif args.material == 'zirconia_2':
            # ZrO2 results (placeholder - would need actual data)
            print("ZrO2 results not available in this demo")
            sys.exit(1)
        else:
            print(f"Unknown material: {args.material}")
            sys.exit(1)
        
        for ne, e in energies.items():
            print(f"  {ne}e: E = {e:.6f} eV")
        
        ne = args.ne_neutral
        bandgap = calculate_bandgap(
            energies[ne - 1], 
            energies[ne], 
            energies[ne + 1]
        )
    
    # Print band gap calculation
    print(f"\n{'=' * 60}")
    print("Band Gap Calculation")
    print(f"{'=' * 60}")
    print("Formula: Eg = E[Ne-1] + E[Ne+1] - 2*E[Ne]")
    print(f"       = {energies[ne-1]:.6f} + {energies[ne+1]:.6f} - 2*{energies[ne]:.6f}")
    print(f"       = {bandgap:.4f} eV")
    
    # Compare with experimental and DFT values
    print(f"\n{'=' * 60}")
    print("Comparison with Reference Values")
    print(f"{'=' * 60}")
    
    exp_bandgap = EXPERIMENTAL_BANDGAPS.get(material_name)
    dft_bandgap = DFT_UV_BANDGAPS.get(material_name)
    
    if exp_bandgap:
        comparison = compare_with_experiment(bandgap, exp_bandgap, dft_bandgap)
        
        print("\nSummary:")
        print(f"  SQD achieves {comparison['sqd_error_percent']:.1f}% error vs experiment")
        if dft_bandgap:
            print(f"  DFT+U+V has {comparison['dft_error_percent']:.1f}% error vs experiment")
            print(f"  SQD reduces error by {comparison['improvement_percent']:.1f}%")
    
    print(f"\n{'=' * 60}")
    print("Calculation Complete")
    print(f"{'=' * 60}")
    
    return bandgap


if __name__ == "__main__":
    main()
