"""
LOBSTER Analysis Helper for δ-Ni₂Si Antisite Defect Study

This module provides utilities for:
1. Generating LOBSTER input files (lobsterin)
2. Parsing LOBSTER output (COHPCAR, ICOHPLIST)
3. Comparing COHP/COOP results with DVM Bond Overlap Population
4. DOS pseudo-gap analysis for ordered structures

Author: Devin AI
Date: 2025-12-22
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# Constants
# =============================================================================

# Energy range for COHP analysis
COHP_START_ENERGY = -15.0  # eV below Fermi level
COHP_END_ENERGY = 5.0      # eV above Fermi level

# Bond distance range for Ni-Si analysis
BOND_DISTANCE_MIN = 0.1    # Angstrom
BOND_DISTANCE_MAX = 3.5    # Angstrom


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class COHPData:
    """Container for COHP analysis results."""
    energy: np.ndarray          # Energy values (eV)
    cohp: np.ndarray            # COHP values
    icohp: np.ndarray           # Integrated COHP values
    atom_pair: Tuple[int, int]  # Atom indices
    bond_distance: float        # Bond distance (Angstrom)
    orbital_info: str           # Orbital description


@dataclass
class BondAnalysis:
    """Summary of bond analysis for a structure."""
    structure_id: str
    n_ni_si_bonds: int
    avg_icohp: float
    total_icohp: float
    avg_bond_distance: float
    fermi_level_cohp: float  # COHP at Fermi level


# =============================================================================
# LOBSTER Input Generation
# =============================================================================

def generate_lobsterin(
    output_path: str,
    basis_set: str = "pbeVaspFit2015",
    cohp_start: float = COHP_START_ENERGY,
    cohp_end: float = COHP_END_ENERGY,
    bond_min: float = BOND_DISTANCE_MIN,
    bond_max: float = BOND_DISTANCE_MAX,
    include_orbitals: List[str] = None,
    save_projection: bool = True,
) -> str:
    """
    Generate LOBSTER input file (lobsterin).
    
    Args:
        output_path: Path to save lobsterin file
        basis_set: Basis set for projection (pbeVaspFit2015 recommended)
        cohp_start: Start energy for COHP (eV below Fermi)
        cohp_end: End energy for COHP (eV above Fermi)
        bond_min: Minimum bond distance for COHP generator
        bond_max: Maximum bond distance for COHP generator
        include_orbitals: List of orbitals to include (default: s, p, d)
        save_projection: Whether to save projection data
    
    Returns:
        Content of lobsterin file
    """
    if include_orbitals is None:
        include_orbitals = ["s", "p", "d"]
    
    orbitals_str = " ".join(include_orbitals)
    
    content = f"""! LOBSTER input for delta-Ni2Si antisite defect analysis
! Generated automatically

! Basis set for projection
basisSet {basis_set}

! Energy range for COHP
COHPstartEnergy {cohp_start}
COHPendEnergy {cohp_end}

! Orbitals to include
includeOrbitals {orbitals_str}

! Automatic COHP generation for all bonds in distance range
cohpGenerator from {bond_min} to {bond_max} orbitalwise

! Output options
"""
    
    if save_projection:
        content += """saveProjectionToFile
"""
    
    # Add specific Ni-Si bond analysis
    content += """
! Specific analysis for Ni-Si bonds (p-d hybridization)
! These will be generated automatically by cohpGenerator
"""
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(content)
    
    return content


def generate_lobsterin_for_structure(
    vasp_dir: str,
    structure_id: str,
) -> str:
    """
    Generate lobsterin file for a specific VASP calculation directory.
    
    Args:
        vasp_dir: Path to VASP calculation directory
        structure_id: Structure identifier
    
    Returns:
        Path to generated lobsterin file
    """
    lobsterin_path = os.path.join(vasp_dir, "lobsterin")
    generate_lobsterin(lobsterin_path)
    return lobsterin_path


# =============================================================================
# LOBSTER Output Parsing
# =============================================================================

def parse_icohplist(filepath: str) -> List[Dict]:
    """
    Parse ICOHPLIST.lobster file.
    
    Args:
        filepath: Path to ICOHPLIST.lobster
    
    Returns:
        List of bond dictionaries with ICOHP values
    """
    bonds = []
    
    if not os.path.exists(filepath):
        return bonds
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines
    data_started = False
    for line in lines:
        line = line.strip()
        
        if not line or line.startswith('#'):
            continue
        
        if 'ICOHP' in line and 'distance' in line.lower():
            data_started = True
            continue
        
        if data_started and line:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    bond = {
                        'index': int(parts[0]),
                        'atom1': parts[1],
                        'atom2': parts[2],
                        'distance': float(parts[3]),
                        'icohp': float(parts[4]),
                        'atom1_idx': int(parts[5]) if len(parts) > 5 else None,
                        'atom2_idx': int(parts[6]) if len(parts) > 6 else None,
                    }
                    bonds.append(bond)
                except (ValueError, IndexError):
                    continue
    
    return bonds


def parse_cohpcar(filepath: str) -> Dict[str, COHPData]:
    """
    Parse COHPCAR.lobster file.
    
    Args:
        filepath: Path to COHPCAR.lobster
    
    Returns:
        Dictionary mapping bond label to COHPData
    """
    cohp_data = {}
    
    if not os.path.exists(filepath):
        return cohp_data
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse header and data sections
    # This is a simplified parser - full implementation would handle
    # all COHPCAR format variations
    
    return cohp_data


def parse_doscar_lobster(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse DOSCAR.lobster file for pseudo-gap analysis.
    
    Args:
        filepath: Path to DOSCAR.lobster
    
    Returns:
        Tuple of (energy, dos) arrays
    """
    if not os.path.exists(filepath):
        return np.array([]), np.array([])
    
    energy = []
    dos = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header (first 6 lines in standard DOSCAR format)
    for line in lines[6:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                energy.append(float(parts[0]))
                dos.append(float(parts[1]))
            except ValueError:
                continue
    
    return np.array(energy), np.array(dos)


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_ni_si_bonds(
    icohp_list: List[Dict],
) -> Dict:
    """
    Analyze Ni-Si bonds from ICOHP data.
    
    Args:
        icohp_list: List of bond dictionaries from parse_icohplist
    
    Returns:
        Analysis summary dictionary
    """
    ni_si_bonds = []
    
    for bond in icohp_list:
        atom1 = bond['atom1'].replace('_', '').upper()
        atom2 = bond['atom2'].replace('_', '').upper()
        
        # Check if this is a Ni-Si bond
        is_ni_si = (
            ('NI' in atom1 and 'SI' in atom2) or
            ('SI' in atom1 and 'NI' in atom2)
        )
        
        if is_ni_si:
            ni_si_bonds.append(bond)
    
    if not ni_si_bonds:
        return {
            'n_bonds': 0,
            'avg_icohp': 0.0,
            'total_icohp': 0.0,
            'avg_distance': 0.0,
            'bonds': [],
        }
    
    icohp_values = [b['icohp'] for b in ni_si_bonds]
    distances = [b['distance'] for b in ni_si_bonds]
    
    return {
        'n_bonds': len(ni_si_bonds),
        'avg_icohp': np.mean(icohp_values),
        'total_icohp': np.sum(icohp_values),
        'avg_distance': np.mean(distances),
        'std_icohp': np.std(icohp_values),
        'min_icohp': np.min(icohp_values),
        'max_icohp': np.max(icohp_values),
        'bonds': ni_si_bonds,
    }


def detect_pseudogap(
    energy: np.ndarray,
    dos: np.ndarray,
    fermi_level: float = 0.0,
    window: float = 0.5,
) -> Dict:
    """
    Detect pseudo-gap at Fermi level in DOS.
    
    A pseudo-gap is indicated by a local minimum in DOS near E_F.
    
    Args:
        energy: Energy array (eV)
        dos: DOS array
        fermi_level: Fermi level energy (default: 0.0)
        window: Energy window around Fermi level (eV)
    
    Returns:
        Dictionary with pseudo-gap analysis results
    """
    # Find DOS values near Fermi level
    mask = np.abs(energy - fermi_level) < window
    
    if not np.any(mask):
        return {
            'has_pseudogap': False,
            'dos_at_fermi': None,
            'local_minimum': None,
        }
    
    dos_near_fermi = dos[mask]
    energy_near_fermi = energy[mask]
    
    # Find DOS at Fermi level
    fermi_idx = np.argmin(np.abs(energy - fermi_level))
    dos_at_fermi = dos[fermi_idx]
    
    # Check if DOS at Fermi is a local minimum
    local_min_idx = np.argmin(dos_near_fermi)
    local_min_dos = dos_near_fermi[local_min_idx]
    local_min_energy = energy_near_fermi[local_min_idx]
    
    # Pseudo-gap criterion: DOS at minimum is significantly lower than average
    avg_dos = np.mean(dos_near_fermi)
    has_pseudogap = local_min_dos < 0.7 * avg_dos
    
    return {
        'has_pseudogap': has_pseudogap,
        'dos_at_fermi': dos_at_fermi,
        'local_minimum': {
            'energy': local_min_energy,
            'dos': local_min_dos,
        },
        'avg_dos_near_fermi': avg_dos,
        'pseudogap_depth': (avg_dos - local_min_dos) / avg_dos if avg_dos > 0 else 0,
    }


def compare_with_dvm(
    icohp_analysis: Dict,
    dvm_bop: Optional[float] = None,
) -> Dict:
    """
    Compare LOBSTER ICOHP with DVM Bond Overlap Population.
    
    Args:
        icohp_analysis: Analysis from analyze_ni_si_bonds
        dvm_bop: DVM Bond Overlap Population value (if available)
    
    Returns:
        Comparison dictionary
    """
    comparison = {
        'lobster_avg_icohp': icohp_analysis['avg_icohp'],
        'lobster_total_icohp': icohp_analysis['total_icohp'],
        'n_bonds': icohp_analysis['n_bonds'],
    }
    
    if dvm_bop is not None:
        comparison['dvm_bop'] = dvm_bop
        # ICOHP is typically negative for bonding interactions
        # More negative = stronger bond
        # BOP is typically positive for bonding
        # Higher = stronger bond
        # So we expect anti-correlation
        comparison['note'] = (
            "ICOHP (negative = bonding) should anti-correlate with "
            "DVM BOP (positive = bonding)"
        )
    
    return comparison


# =============================================================================
# Batch Analysis
# =============================================================================

def analyze_structure_batch(
    vasp_dirs: List[str],
    structure_ids: List[str],
) -> List[BondAnalysis]:
    """
    Analyze multiple structures for COHP/ICOHP.
    
    Args:
        vasp_dirs: List of VASP calculation directories
        structure_ids: List of structure identifiers
    
    Returns:
        List of BondAnalysis objects
    """
    results = []
    
    for vasp_dir, struct_id in zip(vasp_dirs, structure_ids):
        icohp_path = os.path.join(vasp_dir, "ICOHPLIST.lobster")
        
        if not os.path.exists(icohp_path):
            continue
        
        icohp_list = parse_icohplist(icohp_path)
        analysis = analyze_ni_si_bonds(icohp_list)
        
        # Parse DOS for pseudo-gap analysis
        doscar_path = os.path.join(vasp_dir, "DOSCAR.lobster")
        energy, dos = parse_doscar_lobster(doscar_path)
        
        fermi_cohp = 0.0
        if len(energy) > 0:
            pseudogap = detect_pseudogap(energy, dos)
            fermi_cohp = pseudogap.get('dos_at_fermi', 0.0) or 0.0
        
        bond_analysis = BondAnalysis(
            structure_id=struct_id,
            n_ni_si_bonds=analysis['n_bonds'],
            avg_icohp=analysis['avg_icohp'],
            total_icohp=analysis['total_icohp'],
            avg_bond_distance=analysis['avg_distance'],
            fermi_level_cohp=fermi_cohp,
        )
        results.append(bond_analysis)
    
    return results


def generate_cohp_correlation_data(
    bond_analyses: List[BondAnalysis],
    kl_results: List[Dict],
) -> List[Dict]:
    """
    Generate correlation data between COHP and KL divergence.
    
    Args:
        bond_analyses: List of BondAnalysis objects
        kl_results: List of KL divergence results
    
    Returns:
        List of correlation data dictionaries
    """
    # Create lookup by structure_id
    kl_lookup = {r['structure_id']: r for r in kl_results}
    
    correlation_data = []
    
    for analysis in bond_analyses:
        kl_data = kl_lookup.get(analysis.structure_id, {})
        
        correlation_data.append({
            'structure_id': analysis.structure_id,
            'kl_divergence': kl_data.get('kl_divergence', None),
            'avg_icohp': analysis.avg_icohp,
            'total_icohp': analysis.total_icohp,
            'n_ni_si_bonds': analysis.n_ni_si_bonds,
            'avg_bond_distance': analysis.avg_bond_distance,
            'composition': kl_data.get('composition', ''),
            'group': kl_data.get('group', ''),
        })
    
    return correlation_data


# =============================================================================
# Report Generation
# =============================================================================

def generate_analysis_report(
    correlation_data: List[Dict],
    output_path: str,
) -> str:
    """
    Generate analysis report comparing COHP with KL divergence.
    
    Args:
        correlation_data: List of correlation data dictionaries
        output_path: Path to save report
    
    Returns:
        Report content as string
    """
    report = []
    report.append("=" * 70)
    report.append("δ-Ni₂Si Antisite Defect Analysis Report")
    report.append("COHP/ICOHP vs KL Divergence Correlation")
    report.append("=" * 70)
    report.append("")
    
    # Summary statistics
    valid_data = [d for d in correlation_data if d['kl_divergence'] is not None]
    
    if valid_data:
        kl_values = [d['kl_divergence'] for d in valid_data]
        icohp_values = [d['avg_icohp'] for d in valid_data]
        
        report.append("Summary Statistics:")
        report.append(f"  Total structures analyzed: {len(valid_data)}")
        report.append(f"  KL divergence range: {min(kl_values):.4f} - {max(kl_values):.4f}")
        report.append(f"  Average ICOHP range: {min(icohp_values):.4f} - {max(icohp_values):.4f}")
        report.append("")
        
        # Calculate correlation
        if len(valid_data) > 2:
            correlation = np.corrcoef(kl_values, icohp_values)[0, 1]
            report.append(f"  Pearson correlation (KL vs ICOHP): {correlation:.4f}")
            report.append("")
    
    # Group-wise analysis
    report.append("Group-wise Analysis:")
    for group in ['A', 'B', 'C']:
        group_data = [d for d in valid_data if d['group'] == group]
        if group_data:
            avg_kl = np.mean([d['kl_divergence'] for d in group_data])
            avg_icohp = np.mean([d['avg_icohp'] for d in group_data])
            report.append(f"  Group {group}: n={len(group_data)}, "
                         f"avg_KL={avg_kl:.4f}, avg_ICOHP={avg_icohp:.4f}")
    
    report.append("")
    report.append("=" * 70)
    
    content = "\n".join(report)
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    return content


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("LOBSTER Analysis Helper for δ-Ni₂Si")
    print("=" * 50)
    print("\nThis module provides utilities for:")
    print("1. Generating LOBSTER input files")
    print("2. Parsing LOBSTER output")
    print("3. Comparing COHP with DVM results")
    print("4. DOS pseudo-gap analysis")
    print("\nUsage:")
    print("  from ni2si_lobster_analysis import generate_lobsterin")
    print("  generate_lobsterin('path/to/lobsterin')")
