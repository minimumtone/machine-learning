"""
Band Gap Calculation Module

This module calculates the band gap using the SQD method:
Eg = E[Ne-1] + E[Ne+1] - 2*E[Ne]

where Ne is the number of electrons in the neutral system.
"""

import json
import os
from typing import Dict, Any, Optional


def calculate_bandgap(e_ne_minus_1: float, 
                      e_ne: float, 
                      e_ne_plus_1: float) -> float:
    """
    Calculate band gap from total energies.
    
    The band gap is computed as:
    Eg = E[Ne-1] + E[Ne+1] - 2*E[Ne]
    
    This corresponds to the difference between ionization potential (IP)
    and electron affinity (EA):
    Eg = IP - EA = (E[Ne-1] - E[Ne]) - (E[Ne] - E[Ne+1])
    
    Args:
        e_ne_minus_1: Total energy with Ne-1 electrons (cation)
        e_ne: Total energy with Ne electrons (neutral)
        e_ne_plus_1: Total energy with Ne+1 electrons (anion)
    
    Returns:
        bandgap: Band gap in the same units as input energies
    """
    return e_ne_minus_1 + e_ne_plus_1 - 2 * e_ne


def load_sqd_results(results_folder: str, 
                     ne_values: list = None,
                     sampling: str = "hardware") -> Dict[int, float]:
    """
    Load SQD results from result files.
    
    Args:
        results_folder: Path to the folder containing SQD results
        ne_values: List of electron numbers to load (default: [Ne-1, Ne, Ne+1])
        sampling: Sampling method ('hardware', 'ffsim', etc.)
    
    Returns:
        energies: Dictionary mapping electron number to SQD energy
    """
    energies = {}
    
    for ne in ne_values:
        ne_folder = os.path.join(results_folder, f"{ne}e", f"results_sqd_{sampling}", "dicts")
        
        if not os.path.exists(ne_folder):
            raise FileNotFoundError(f"Results folder not found: {ne_folder}")
        
        # Find the result file with highest samples_per_batch
        files = sorted([f for f in os.listdir(ne_folder) if f.startswith("results_dict_")])
        if not files:
            raise FileNotFoundError(f"No result files found in {ne_folder}")
        
        latest_file = files[-1]
        with open(os.path.join(ne_folder, latest_file)) as f:
            data = json.load(f)
        
        energies[ne] = data['sqd_energy']
        print(f"Loaded {ne}e: E = {energies[ne]:.6f} eV (file: {latest_file})")
    
    return energies


def compute_bandgap_from_results(results_folder: str,
                                 ne_neutral: int,
                                 sampling: str = "hardware") -> Dict[str, Any]:
    """
    Compute band gap from SQD results.
    
    Args:
        results_folder: Path to the folder containing SQD results
        ne_neutral: Number of electrons in the neutral system
        sampling: Sampling method ('hardware', 'ffsim', etc.)
    
    Returns:
        results: Dictionary containing energies and band gap
    """
    ne_values = [ne_neutral - 1, ne_neutral, ne_neutral + 1]
    
    energies = load_sqd_results(results_folder, ne_values, sampling)
    
    e_minus = energies[ne_neutral - 1]
    e_neutral = energies[ne_neutral]
    e_plus = energies[ne_neutral + 1]
    
    bandgap = calculate_bandgap(e_minus, e_neutral, e_plus)
    
    results = {
        'E_Ne-1': e_minus,
        'E_Ne': e_neutral,
        'E_Ne+1': e_plus,
        'bandgap': bandgap,
        'ne_neutral': ne_neutral,
        'sampling': sampling
    }
    
    print(f"\n=== Band Gap Calculation ===")
    print(f"E[Ne-1] = E[{ne_neutral-1}e] = {e_minus:.6f} eV")
    print(f"E[Ne]   = E[{ne_neutral}e] = {e_neutral:.6f} eV")
    print(f"E[Ne+1] = E[{ne_neutral+1}e] = {e_plus:.6f} eV")
    print(f"\nBand Gap = E[Ne-1] + E[Ne+1] - 2*E[Ne]")
    print(f"         = {e_minus:.6f} + {e_plus:.6f} - 2*{e_neutral:.6f}")
    print(f"         = {bandgap:.4f} eV")
    
    return results


def compare_with_experiment(sqd_bandgap: float,
                           experimental_bandgap: float,
                           dft_bandgap: Optional[float] = None) -> Dict[str, float]:
    """
    Compare SQD band gap with experimental and DFT values.
    
    Args:
        sqd_bandgap: Band gap from SQD calculation
        experimental_bandgap: Experimental band gap
        dft_bandgap: Band gap from DFT+U+V (optional)
    
    Returns:
        comparison: Dictionary with errors and improvements
    """
    sqd_error = abs(sqd_bandgap - experimental_bandgap)
    sqd_error_percent = 100 * sqd_error / experimental_bandgap
    
    comparison = {
        'sqd_bandgap': sqd_bandgap,
        'experimental_bandgap': experimental_bandgap,
        'sqd_error': sqd_error,
        'sqd_error_percent': sqd_error_percent
    }
    
    print(f"\n=== Comparison with Experiment ===")
    print(f"SQD Band Gap:          {sqd_bandgap:.4f} eV")
    print(f"Experimental:          {experimental_bandgap:.4f} eV")
    print(f"SQD Error:             {sqd_error:.4f} eV ({sqd_error_percent:.1f}%)")
    
    if dft_bandgap is not None:
        dft_error = abs(dft_bandgap - experimental_bandgap)
        dft_error_percent = 100 * dft_error / experimental_bandgap
        improvement = dft_error - sqd_error
        improvement_percent = 100 * improvement / dft_error if dft_error > 0 else 0
        
        comparison['dft_bandgap'] = dft_bandgap
        comparison['dft_error'] = dft_error
        comparison['dft_error_percent'] = dft_error_percent
        comparison['improvement'] = improvement
        comparison['improvement_percent'] = improvement_percent
        
        print(f"DFT+U+V Band Gap:      {dft_bandgap:.4f} eV")
        print(f"DFT+U+V Error:         {dft_error:.4f} eV ({dft_error_percent:.1f}%)")
        print(f"SQD Improvement:       {improvement:.4f} eV ({improvement_percent:.1f}% reduction)")
    
    return comparison


# Reference experimental values (in eV)
EXPERIMENTAL_BANDGAPS = {
    'HfO2': 5.7,   # Hafnium dioxide
    'ZrO2': 5.8,   # Zirconium dioxide
}

# Reference DFT+U+V values (in eV, from paper)
DFT_UV_BANDGAPS = {
    'HfO2': 4.5,
    'ZrO2': 4.3,
}
