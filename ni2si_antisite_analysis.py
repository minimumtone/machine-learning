"""
delta-Ni2Si Antisite Defect Formation Energy Analysis

δ-Ni₂Si（Pnma構造）における非化学量論組成のSRO解析

This module implements a comprehensive workflow for:
1. Crystal structure generation for δ-Ni₂Si (Pnma, #62)
2. Structure sampling (random, SA-optimized, specific defect configurations)
3. VASP input file generation
4. KL divergence calculation for local environment analysis
5. DVM comparison analysis framework

Based on the detailed instruction document for antisite defect formation energy calculations.

Author: Devin AI
Date: 2025-12-22
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import random
from collections import defaultdict
import warnings


# =============================================================================
# Constants and Crystal Structure Parameters
# =============================================================================

# Lattice constants for δ-Ni₂Si (Pnma)
LATTICE_A = 5.00  # Angstrom
LATTICE_B = 3.73  # Angstrom
LATTICE_C = 7.04  # Angstrom

# Supercell dimensions
SUPERCELL_X = 2
SUPERCELL_Y = 2
SUPERCELL_Z = 1

# Wyckoff 4c positions (x, 1/4, z) for Pnma
WYCKOFF_4C = {
    "Ni1": (0.038, 0.250, 0.218),
    "Ni2": (0.183, 0.250, 0.561),
    "Si1": (0.712, 0.250, 0.611),
}

# Pair potential parameters for simulated annealing
PAIR_POTENTIAL_EPSILON = {
    ("Ni", "Si"): -1.0,  # Attractive
    ("Si", "Ni"): -1.0,  # Symmetric
    ("Ni", "Ni"): 0.5,   # Repulsive
    ("Si", "Si"): 0.8,   # Repulsive
}

# Default pair potential exponent (H = Σ ε_αβ/r_ij^k)
DEFAULT_PAIR_POTENTIAL_K = 1

# Cutoff distances
SA_CUTOFF = 3.5  # Angstrom for SA pair potential
KL_CUTOFF = 3.2  # Angstrom for KL divergence local environment


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AtomSite:
    """Represents a single atomic site in the crystal structure."""
    index: int
    sublattice: str  # "Ni1", "Ni2", or "Si1"
    ideal_species: str  # "Ni" or "Si" (what should be there stoichiometrically)
    frac_coords: Tuple[float, float, float]
    cart_coords: Tuple[float, float, float]
    image: Tuple[int, int, int]  # Which supercell tile (i, j, k)
    wyckoff_index: int  # Which of the 4 equivalent positions (0-3)


@dataclass
class CrystalStructure:
    """Represents the full crystal structure with site mapping."""
    lattice_vectors: np.ndarray  # 3x3 matrix
    sites: List[AtomSite]
    supercell_dim: Tuple[int, int, int]
    space_group: str = "Pnma"
    space_group_number: int = 62
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "lattice_vectors": self.lattice_vectors.tolist(),
            "supercell_dim": list(self.supercell_dim),
            "space_group": self.space_group,
            "space_group_number": self.space_group_number,
            "n_atoms": len(self.sites),
            "sites": [
                {
                    "index": s.index,
                    "sublattice": s.sublattice,
                    "ideal_species": s.ideal_species,
                    "frac_coords": list(s.frac_coords),
                    "cart_coords": list(s.cart_coords),
                    "image": list(s.image),
                    "wyckoff_index": s.wyckoff_index,
                }
                for s in self.sites
            ],
            "sublattice_indices": self.get_sublattice_indices(),
        }
    
    def get_sublattice_indices(self) -> Dict[str, List[int]]:
        """Get indices grouped by sublattice type."""
        result = {"Ni1": [], "Ni2": [], "Si1": []}
        for site in self.sites:
            result[site.sublattice].append(site.index)
        return result


@dataclass
class Configuration:
    """Represents a specific atomic configuration (occupancy pattern)."""
    structure_id: str
    group: str  # "A", "B", or "C"
    composition: str  # e.g., "Ni32Si16"
    occupancy: List[str]  # Species at each site ("Ni" or "Si")
    n_ni: int
    n_si: int
    metadata: Dict = field(default_factory=dict)
    
    def get_occupancy_hash(self) -> str:
        """Get a unique hash for this occupancy pattern."""
        return "".join(["1" if s == "Ni" else "0" for s in self.occupancy])


# =============================================================================
# Pnma Space Group Operations
# =============================================================================

def apply_pnma_4c_operations(x: float, z: float) -> List[Tuple[float, float, float]]:
    """
    Apply Pnma space group operations to generate 4c Wyckoff equivalent positions.
    
    For Wyckoff position 4c in Pnma (#62), the general position (x, 1/4, z)
    generates 4 equivalent positions through the following operations:
    
    1. (x, 1/4, z)
    2. (1/2-x, 3/4, 1/2+z)
    3. (-x, 3/4, -z)
    4. (1/2+x, 1/4, 1/2-z)
    
    All coordinates are wrapped to [0, 1) range.
    
    Args:
        x: x-coordinate of the Wyckoff position
        z: z-coordinate of the Wyckoff position
    
    Returns:
        List of 4 equivalent positions as (x, y, z) tuples
    """
    positions = [
        (x, 0.25, z),
        (0.5 - x, 0.75, 0.5 + z),
        (-x, 0.75, -z),
        (0.5 + x, 0.25, 0.5 - z),
    ]
    
    # Wrap to [0, 1) range
    wrapped = []
    for pos in positions:
        wrapped_pos = tuple(p % 1.0 for p in pos)
        wrapped.append(wrapped_pos)
    
    return wrapped


def generate_unit_cell_sites() -> List[Tuple[str, str, Tuple[float, float, float], int]]:
    """
    Generate all atomic sites in the unit cell.
    
    Returns:
        List of (sublattice, ideal_species, frac_coords, wyckoff_index) tuples
    """
    sites = []
    
    for sublattice, (x, y, z) in WYCKOFF_4C.items():
        # Determine ideal species from sublattice name
        ideal_species = "Ni" if sublattice.startswith("Ni") else "Si"
        
        # Generate 4 equivalent positions
        equiv_positions = apply_pnma_4c_operations(x, z)
        
        for wyckoff_idx, pos in enumerate(equiv_positions):
            sites.append((sublattice, ideal_species, pos, wyckoff_idx))
    
    return sites


def frac_to_cart(frac_coords: Tuple[float, float, float], 
                 lattice_vectors: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert fractional coordinates to Cartesian coordinates.
    
    Args:
        frac_coords: Fractional coordinates (a, b, c)
        lattice_vectors: 3x3 matrix of lattice vectors (rows are vectors)
    
    Returns:
        Cartesian coordinates (x, y, z)
    """
    frac = np.array(frac_coords)
    cart = frac @ lattice_vectors
    return tuple(cart)


# =============================================================================
# Crystal Structure Generation
# =============================================================================

def generate_ni2si_supercell(
    supercell_x: int = SUPERCELL_X,
    supercell_y: int = SUPERCELL_Y,
    supercell_z: int = SUPERCELL_Z,
) -> CrystalStructure:
    """
    Generate the δ-Ni₂Si supercell structure with complete site mapping.
    
    This function creates a 2×2×1 supercell (default) of δ-Ni₂Si with:
    - 48 atoms total (32 Ni + 16 Si for stoichiometric composition)
    - Complete site mapping including sublattice type, ideal species, and coordinates
    
    Args:
        supercell_x: Number of unit cells in x direction
        supercell_y: Number of unit cells in y direction
        supercell_z: Number of unit cells in z direction
    
    Returns:
        CrystalStructure object with all site information
    """
    # Calculate supercell lattice vectors
    lattice_vectors = np.array([
        [LATTICE_A * supercell_x, 0, 0],
        [0, LATTICE_B * supercell_y, 0],
        [0, 0, LATTICE_C * supercell_z],
    ])
    
    # Generate unit cell sites
    unit_cell_sites = generate_unit_cell_sites()
    
    # Generate supercell sites
    sites = []
    site_index = 0
    
    for i in range(supercell_x):
        for j in range(supercell_y):
            for k in range(supercell_z):
                for sublattice, ideal_species, frac, wyckoff_idx in unit_cell_sites:
                    # Calculate supercell fractional coordinates
                    supercell_frac = (
                        (frac[0] + i) / supercell_x,
                        (frac[1] + j) / supercell_y,
                        (frac[2] + k) / supercell_z,
                    )
                    
                    # Calculate Cartesian coordinates
                    cart = frac_to_cart(supercell_frac, lattice_vectors)
                    
                    site = AtomSite(
                        index=site_index,
                        sublattice=sublattice,
                        ideal_species=ideal_species,
                        frac_coords=supercell_frac,
                        cart_coords=cart,
                        image=(i, j, k),
                        wyckoff_index=wyckoff_idx,
                    )
                    sites.append(site)
                    site_index += 1
    
    return CrystalStructure(
        lattice_vectors=lattice_vectors,
        sites=sites,
        supercell_dim=(supercell_x, supercell_y, supercell_z),
    )


def save_site_mapping(structure: CrystalStructure, filepath: str) -> None:
    """
    Save the site mapping to a JSON file.
    
    Args:
        structure: CrystalStructure object
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(structure.to_dict(), f, indent=2, ensure_ascii=False)


# =============================================================================
# Distance Calculations with Periodic Boundary Conditions
# =============================================================================

def calculate_distance_pbc(
    pos1: np.ndarray,
    pos2: np.ndarray,
    lattice_vectors: np.ndarray,
) -> float:
    """
    Calculate the minimum image distance between two positions with PBC.
    
    Args:
        pos1: Cartesian coordinates of first position
        pos2: Cartesian coordinates of second position
        lattice_vectors: 3x3 matrix of lattice vectors
    
    Returns:
        Minimum image distance
    """
    # Calculate inverse lattice matrix for fractional conversion
    inv_lattice = np.linalg.inv(lattice_vectors)
    
    # Convert to fractional coordinates
    diff_cart = pos2 - pos1
    diff_frac = diff_cart @ inv_lattice
    
    # Apply minimum image convention
    diff_frac = diff_frac - np.round(diff_frac)
    
    # Convert back to Cartesian
    diff_cart_min = diff_frac @ lattice_vectors
    
    return np.linalg.norm(diff_cart_min)


def build_neighbor_list(
    structure: CrystalStructure,
    cutoff: float,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build neighbor list for all sites within cutoff distance.
    
    Args:
        structure: CrystalStructure object
        cutoff: Cutoff distance in Angstrom
    
    Returns:
        Dictionary mapping site index to list of (neighbor_index, distance) tuples
    """
    n_atoms = len(structure.sites)
    neighbor_list = {i: [] for i in range(n_atoms)}
    
    for i in range(n_atoms):
        pos_i = np.array(structure.sites[i].cart_coords)
        for j in range(i + 1, n_atoms):
            pos_j = np.array(structure.sites[j].cart_coords)
            
            dist = calculate_distance_pbc(pos_i, pos_j, structure.lattice_vectors)
            
            if dist < cutoff:
                neighbor_list[i].append((j, dist))
                neighbor_list[j].append((i, dist))
    
    return neighbor_list


def build_pair_list_with_weights(
    structure: CrystalStructure,
    cutoff: float,
    k: float = DEFAULT_PAIR_POTENTIAL_K,
) -> List[Tuple[int, int, float]]:
    """
    Build pair list with distance-based weights for SA Hamiltonian.
    
    Args:
        structure: CrystalStructure object
        cutoff: Cutoff distance in Angstrom
        k: Exponent for distance weighting (1/r^k)
    
    Returns:
        List of (i, j, weight) tuples where weight = 1/r^k
    """
    pairs = []
    n_atoms = len(structure.sites)
    
    for i in range(n_atoms):
        pos_i = np.array(structure.sites[i].cart_coords)
        for j in range(i + 1, n_atoms):
            pos_j = np.array(structure.sites[j].cart_coords)
            
            dist = calculate_distance_pbc(pos_i, pos_j, structure.lattice_vectors)
            
            if dist < cutoff and dist > 0.1:  # Avoid division by zero
                weight = 1.0 / (dist ** k)
                pairs.append((i, j, weight))
    
    return pairs


# =============================================================================
# Structure Sampling: Group A (Random Baseline)
# =============================================================================

def generate_random_configuration(
    structure: CrystalStructure,
    n_ni: int,
    n_si: int,
    random_seed: Optional[int] = None,
) -> Configuration:
    """
    Generate a random configuration with specified composition.
    
    Args:
        structure: CrystalStructure object
        n_ni: Number of Ni atoms
        n_si: Number of Si atoms
        random_seed: Random seed for reproducibility
    
    Returns:
        Configuration object with random occupancy
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    n_atoms = len(structure.sites)
    if n_ni + n_si != n_atoms:
        raise ValueError(f"n_ni ({n_ni}) + n_si ({n_si}) must equal n_atoms ({n_atoms})")
    
    # Create occupancy list
    occupancy = ["Ni"] * n_ni + ["Si"] * n_si
    random.shuffle(occupancy)
    
    return Configuration(
        structure_id="",  # Will be set later
        group="A",
        composition=f"Ni{n_ni}Si{n_si}",
        occupancy=occupancy,
        n_ni=n_ni,
        n_si=n_si,
    )


def generate_group_a_structures(
    structure: CrystalStructure,
    base_seed: int = 42,
) -> List[Configuration]:
    """
    Generate Group A: 100 random baseline structures.
    
    Composition distribution:
    - Ni₃₂Si₁₆: 40 structures (stoichiometric)
    - Ni₃₄Si₁₄: 30 structures (Ni-rich)
    - Ni₃₀Si₁₈: 30 structures (Si-rich)
    
    Args:
        structure: CrystalStructure object
        base_seed: Base random seed
    
    Returns:
        List of 100 Configuration objects
    """
    configurations = []
    
    # Composition specifications: (n_ni, n_si, count)
    compositions = [
        (32, 16, 40),  # Stoichiometric
        (34, 14, 30),  # Ni-rich
        (30, 18, 30),  # Si-rich
    ]
    
    config_idx = 0
    for n_ni, n_si, count in compositions:
        for i in range(count):
            seed = base_seed + config_idx
            config = generate_random_configuration(structure, n_ni, n_si, seed)
            config.structure_id = f"A_{config_idx:03d}"
            config.metadata["seed"] = seed
            configurations.append(config)
            config_idx += 1
    
    return configurations


# =============================================================================
# Structure Sampling: Group B (Simulated Annealing)
# =============================================================================

def calculate_hamiltonian(
    occupancy: List[str],
    pair_list: List[Tuple[int, int, float]],
) -> float:
    """
    Calculate the pair potential Hamiltonian.
    
    H = Σ_{i<j} ε_αβ * w_ij
    
    where w_ij = 1/r_ij^k (pre-calculated in pair_list)
    
    Args:
        occupancy: List of species at each site
        pair_list: List of (i, j, weight) tuples
    
    Returns:
        Hamiltonian value
    """
    H = 0.0
    for i, j, weight in pair_list:
        species_i = occupancy[i]
        species_j = occupancy[j]
        epsilon = PAIR_POTENTIAL_EPSILON.get((species_i, species_j), 0.0)
        H += epsilon * weight
    
    return H


def calculate_delta_hamiltonian(
    occupancy: List[str],
    pair_list: List[Tuple[int, int, float]],
    swap_i: int,
    swap_j: int,
) -> float:
    """
    Calculate the change in Hamiltonian for swapping two atoms.
    
    This is more efficient than recalculating the full Hamiltonian.
    
    Args:
        occupancy: Current occupancy list
        pair_list: List of (i, j, weight) tuples
        swap_i: First atom index to swap
        swap_j: Second atom index to swap
    
    Returns:
        Change in Hamiltonian (ΔH)
    """
    if occupancy[swap_i] == occupancy[swap_j]:
        return 0.0  # No change if same species
    
    delta_H = 0.0
    
    # Find all pairs involving swap_i or swap_j
    for i, j, weight in pair_list:
        if i == swap_i or j == swap_i or i == swap_j or j == swap_j:
            # Calculate contribution before swap
            species_i = occupancy[i]
            species_j = occupancy[j]
            epsilon_before = PAIR_POTENTIAL_EPSILON.get((species_i, species_j), 0.0)
            
            # Calculate contribution after swap
            new_species_i = species_i
            new_species_j = species_j
            if i == swap_i:
                new_species_i = occupancy[swap_j]
            elif i == swap_j:
                new_species_i = occupancy[swap_i]
            if j == swap_i:
                new_species_j = occupancy[swap_j]
            elif j == swap_j:
                new_species_j = occupancy[swap_i]
            
            epsilon_after = PAIR_POTENTIAL_EPSILON.get((new_species_i, new_species_j), 0.0)
            
            delta_H += (epsilon_after - epsilon_before) * weight
    
    return delta_H


def simulated_annealing(
    structure: CrystalStructure,
    n_ni: int,
    n_si: int,
    pair_list: List[Tuple[int, int, float]],
    T_initial: float = 10.0,
    T_final: float = 0.01,
    cooling_rate: float = 0.95,
    steps_per_temp: int = 100,
    random_seed: Optional[int] = None,
    collect_trajectory: bool = True,
) -> Tuple[Configuration, List[Tuple[float, Configuration]]]:
    """
    Perform simulated annealing to find low-energy configurations.
    
    Args:
        structure: CrystalStructure object
        n_ni: Number of Ni atoms
        n_si: Number of Si atoms
        pair_list: Pre-calculated pair list with weights
        T_initial: Initial temperature
        T_final: Final temperature
        cooling_rate: Geometric cooling factor (T_new = T * cooling_rate)
        steps_per_temp: Number of swap attempts per temperature
        random_seed: Random seed for reproducibility
        collect_trajectory: Whether to collect intermediate configurations
    
    Returns:
        Tuple of (final_configuration, trajectory)
        trajectory is a list of (H, Configuration) tuples
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    n_atoms = len(structure.sites)
    
    # Initialize random configuration
    occupancy = ["Ni"] * n_ni + ["Si"] * n_si
    random.shuffle(occupancy)
    
    # Calculate initial Hamiltonian
    H_current = calculate_hamiltonian(occupancy, pair_list)
    
    # Track best configuration
    best_occupancy = occupancy.copy()
    best_H = H_current
    
    # Trajectory for collecting intermediate structures
    trajectory = []
    
    # Get indices of Ni and Si atoms for efficient swapping
    T = T_initial
    
    while T > T_final:
        for _ in range(steps_per_temp):
            # Find Ni and Si atoms for potential swap
            ni_indices = [i for i, s in enumerate(occupancy) if s == "Ni"]
            si_indices = [i for i, s in enumerate(occupancy) if s == "Si"]
            
            if not ni_indices or not si_indices:
                continue
            
            # Select random pair to swap
            swap_i = random.choice(ni_indices)
            swap_j = random.choice(si_indices)
            
            # Calculate energy change
            delta_H = calculate_delta_hamiltonian(occupancy, pair_list, swap_i, swap_j)
            
            # Metropolis criterion
            if delta_H < 0 or random.random() < np.exp(-delta_H / T):
                # Accept swap
                occupancy[swap_i], occupancy[swap_j] = occupancy[swap_j], occupancy[swap_i]
                H_current += delta_H
                
                # Update best if improved
                if H_current < best_H:
                    best_occupancy = occupancy.copy()
                    best_H = H_current
        
        # Collect trajectory point
        if collect_trajectory:
            config = Configuration(
                structure_id="",
                group="B",
                composition=f"Ni{n_ni}Si{n_si}",
                occupancy=occupancy.copy(),
                n_ni=n_ni,
                n_si=n_si,
                metadata={"temperature": T, "hamiltonian": H_current},
            )
            trajectory.append((H_current, config))
        
        # Cool down
        T *= cooling_rate
    
    # Create final configuration
    final_config = Configuration(
        structure_id="",
        group="B",
        composition=f"Ni{n_ni}Si{n_si}",
        occupancy=best_occupancy,
        n_ni=n_ni,
        n_si=n_si,
        metadata={"final_hamiltonian": best_H},
    )
    
    return final_config, trajectory


def generate_group_b_structures(
    structure: CrystalStructure,
    n_structures: int = 100,
    base_seed: int = 1000,
    k: float = DEFAULT_PAIR_POTENTIAL_K,
) -> List[Configuration]:
    """
    Generate Group B: 100 SA-optimized structures.
    
    Samples structures at different H values during cooling process
    across different compositions.
    
    Args:
        structure: CrystalStructure object
        n_structures: Total number of structures to generate
        base_seed: Base random seed
        k: Pair potential exponent
    
    Returns:
        List of Configuration objects
    """
    # Build pair list
    pair_list = build_pair_list_with_weights(structure, SA_CUTOFF, k)
    
    configurations = []
    seen_hashes: Set[str] = set()
    
    # Composition specifications: (n_ni, n_si, target_count)
    compositions = [
        (32, 16, 40),  # Stoichiometric
        (34, 14, 30),  # Ni-rich
        (30, 18, 30),  # Si-rich
    ]
    
    config_idx = 0
    
    for n_ni, n_si, target_count in compositions:
        collected = 0
        run_idx = 0
        
        while collected < target_count and run_idx < target_count * 10:
            seed = base_seed + config_idx * 100 + run_idx
            
            # Run SA
            final_config, trajectory = simulated_annealing(
                structure=structure,
                n_ni=n_ni,
                n_si=n_si,
                pair_list=pair_list,
                random_seed=seed,
                collect_trajectory=True,
            )
            
            # Sample from trajectory at different H values
            if trajectory:
                # Sort by H and sample evenly
                trajectory.sort(key=lambda x: x[0])
                n_samples = min(3, target_count - collected)
                
                # Sample from different parts of the trajectory
                indices = np.linspace(0, len(trajectory) - 1, n_samples + 2, dtype=int)[1:-1]
                
                for idx in indices:
                    if collected >= target_count:
                        break
                    
                    H, config = trajectory[idx]
                    config_hash = config.get_occupancy_hash()
                    
                    if config_hash not in seen_hashes:
                        seen_hashes.add(config_hash)
                        config.structure_id = f"B_{config_idx:03d}"
                        config.metadata["sa_seed"] = seed
                        config.metadata["trajectory_index"] = int(idx)
                        configurations.append(config)
                        collected += 1
                        config_idx += 1
            
            run_idx += 1
    
    return configurations[:n_structures]


# =============================================================================
# Structure Sampling: Group C (Specific Defect Configurations)
# =============================================================================

def generate_site_specific_substitution(
    structure: CrystalStructure,
    target_sublattice: str,
    n_substitutions: int,
    random_seed: Optional[int] = None,
) -> Configuration:
    """
    Generate a configuration with substitutions only on a specific sublattice.
    
    Args:
        structure: CrystalStructure object
        target_sublattice: "Ni1", "Ni2", or "Si1"
        n_substitutions: Number of atoms to substitute
        random_seed: Random seed for reproducibility
    
    Returns:
        Configuration object
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Start with stoichiometric occupancy
    occupancy = [site.ideal_species for site in structure.sites]
    
    # Get indices of target sublattice
    sublattice_indices = structure.get_sublattice_indices()[target_sublattice]
    
    if n_substitutions > len(sublattice_indices):
        raise ValueError(
            f"Cannot substitute {n_substitutions} atoms on {target_sublattice} "
            f"(only {len(sublattice_indices)} sites available)"
        )
    
    # Select sites to substitute
    sites_to_substitute = random.sample(sublattice_indices, n_substitutions)
    
    # Perform substitution
    for idx in sites_to_substitute:
        if occupancy[idx] == "Ni":
            occupancy[idx] = "Si"
        else:
            occupancy[idx] = "Ni"
    
    # Count final composition
    n_ni = sum(1 for s in occupancy if s == "Ni")
    n_si = sum(1 for s in occupancy if s == "Si")
    
    return Configuration(
        structure_id="",
        group="C",
        composition=f"Ni{n_ni}Si{n_si}",
        occupancy=occupancy,
        n_ni=n_ni,
        n_si=n_si,
        metadata={
            "target_sublattice": target_sublattice,
            "n_substitutions": n_substitutions,
            "substituted_sites": sites_to_substitute,
        },
    )


def generate_group_c_structures(
    structure: CrystalStructure,
    base_seed: int = 2000,
) -> List[Configuration]:
    """
    Generate Group C: 56 specific defect configurations.
    
    Creates configurations with site-specific substitutions for DVM comparison:
    - Ni1-only substitutions (various counts): 4 counts × 4 trials = 16
    - Ni2-only substitutions (various counts): 4 counts × 4 trials = 16
    - Si1-only substitutions (various counts): 4 counts × 3 trials = 12
    - Mixed Ni1+Ni2 substitutions: 4 counts × 3 trials = 12
    Total: 56 structures
    
    Args:
        structure: CrystalStructure object
        base_seed: Base random seed
    
    Returns:
        List of Configuration objects
    """
    configurations = []
    config_idx = 0
    
    # Get sublattice sizes
    sublattice_indices = structure.get_sublattice_indices()
    ni1_size = len(sublattice_indices["Ni1"])  # Should be 16
    ni2_size = len(sublattice_indices["Ni2"])  # Should be 16
    si1_size = len(sublattice_indices["Si1"])  # Should be 16
    
    # Ni1-only substitutions (Ni -> Si): 1, 2, 4, 8 substitutions × 4 random samples = 16
    for n_sub in [1, 2, 4, 8]:
        for trial in range(4):
            seed = base_seed + config_idx
            config = generate_site_specific_substitution(
                structure, "Ni1", n_sub, seed
            )
            config.structure_id = f"C_{config_idx:03d}"
            config.metadata["trial"] = trial
            config.metadata["defect_type"] = "Ni1_antisite"
            configurations.append(config)
            config_idx += 1
    
    # Ni2-only substitutions (Ni -> Si): 1, 2, 4, 8 substitutions × 4 random samples = 16
    for n_sub in [1, 2, 4, 8]:
        for trial in range(4):
            seed = base_seed + config_idx
            config = generate_site_specific_substitution(
                structure, "Ni2", n_sub, seed
            )
            config.structure_id = f"C_{config_idx:03d}"
            config.metadata["trial"] = trial
            config.metadata["defect_type"] = "Ni2_antisite"
            configurations.append(config)
            config_idx += 1
    
    # Si1-only substitutions (Si -> Ni): 1, 2, 4, 8 substitutions × 3 random samples = 12
    for n_sub in [1, 2, 4, 8]:
        for trial in range(3):
            seed = base_seed + config_idx
            config = generate_site_specific_substitution(
                structure, "Si1", n_sub, seed
            )
            config.structure_id = f"C_{config_idx:03d}"
            config.metadata["trial"] = trial
            config.metadata["defect_type"] = "Si1_antisite"
            configurations.append(config)
            config_idx += 1
    
    # Mixed Ni1+Ni2 substitutions: equal substitutions on both sublattices
    # 1, 2, 4, 8 total substitutions (split between Ni1 and Ni2) × 3 trials = 12
    for n_sub_total in [2, 4, 6, 8]:
        n_sub_each = n_sub_total // 2
        for trial in range(3):
            seed = base_seed + config_idx
            random.seed(seed)
            
            # Start with stoichiometric occupancy
            occupancy = [site.ideal_species for site in structure.sites]
            
            # Substitute on Ni1
            ni1_indices = sublattice_indices["Ni1"]
            ni1_to_sub = random.sample(ni1_indices, n_sub_each)
            for idx in ni1_to_sub:
                occupancy[idx] = "Si"
            
            # Substitute on Ni2
            ni2_indices = sublattice_indices["Ni2"]
            ni2_to_sub = random.sample(ni2_indices, n_sub_each)
            for idx in ni2_to_sub:
                occupancy[idx] = "Si"
            
            n_ni = sum(1 for s in occupancy if s == "Ni")
            n_si = sum(1 for s in occupancy if s == "Si")
            
            config = Configuration(
                structure_id=f"C_{config_idx:03d}",
                group="C",
                composition=f"Ni{n_ni}Si{n_si}",
                occupancy=occupancy,
                n_ni=n_ni,
                n_si=n_si,
                metadata={
                    "trial": trial,
                    "defect_type": "mixed_Ni1_Ni2_antisite",
                    "n_sub_ni1": n_sub_each,
                    "n_sub_ni2": n_sub_each,
                },
            )
            configurations.append(config)
            config_idx += 1
    
    return configurations


# =============================================================================
# VASP Input File Generation
# =============================================================================

def generate_poscar(
    structure: CrystalStructure,
    config: Configuration,
    comment: str = "",
) -> str:
    """
    Generate VASP POSCAR file content.
    
    Atoms are sorted with Ni first, then Si (to match MAGMOM specification).
    
    Args:
        structure: CrystalStructure object
        config: Configuration object with occupancy
        comment: Comment line for POSCAR
    
    Returns:
        POSCAR file content as string
    """
    lines = []
    
    # Comment line
    if not comment:
        comment = f"{config.structure_id} {config.composition}"
    lines.append(comment)
    
    # Scale factor
    lines.append("1.0")
    
    # Lattice vectors
    for vec in structure.lattice_vectors:
        lines.append(f"  {vec[0]:16.10f}  {vec[1]:16.10f}  {vec[2]:16.10f}")
    
    # Element symbols and counts (Ni first, then Si)
    lines.append("  Ni  Si")
    lines.append(f"  {config.n_ni}  {config.n_si}")
    
    # Coordinate type
    lines.append("Direct")
    
    # Sort atoms: Ni first, then Si
    ni_coords = []
    si_coords = []
    
    for i, site in enumerate(structure.sites):
        if config.occupancy[i] == "Ni":
            ni_coords.append(site.frac_coords)
        else:
            si_coords.append(site.frac_coords)
    
    # Write Ni coordinates
    for frac in ni_coords:
        lines.append(f"  {frac[0]:16.10f}  {frac[1]:16.10f}  {frac[2]:16.10f}")
    
    # Write Si coordinates
    for frac in si_coords:
        lines.append(f"  {frac[0]:16.10f}  {frac[1]:16.10f}  {frac[2]:16.10f}")
    
    return "\n".join(lines)


def generate_incar(config: Configuration) -> str:
    """
    Generate VASP INCAR file content.
    
    Uses the parameters specified in the instruction document.
    
    Args:
        config: Configuration object (for MAGMOM specification)
    
    Returns:
        INCAR file content as string
    """
    # MAGMOM: Ni atoms get 0.6, Si atoms get 0.0
    # Since POSCAR has Ni first, then Si
    magmom = f"{config.n_ni}*0.6 {config.n_si}*0.0"
    
    incar_content = f"""# VASP INCAR for delta-Ni2Si antisite defect calculations
# Structure: {config.structure_id} ({config.composition})

# Precision settings
PREC   = Accurate
ENCUT  = 520
EDIFF  = 1E-6
EDIFFG = -0.02

# Electronic structure
ISMEAR = 0
SIGMA  = 0.05
LREAL  = Auto

# Magnetic settings (required for Ni)
ISPIN  = 2
MAGMOM = {magmom}

# Output for post-processing (LOBSTER, etc.)
LWAVE  = .TRUE.
LCHARG = .TRUE.

# Parallelization (adjust based on cluster)
NCORE  = 4

# Ionic relaxation
IBRION = 2
NSW    = 100
ISIF   = 2
"""
    return incar_content


def generate_kpoints(mesh: Tuple[int, int, int] = (4, 4, 4)) -> str:
    """
    Generate VASP KPOINTS file content.
    
    Args:
        mesh: K-point mesh dimensions
    
    Returns:
        KPOINTS file content as string
    """
    return f"""Automatic mesh
0
Gamma
  {mesh[0]}  {mesh[1]}  {mesh[2]}
  0  0  0
"""


# =============================================================================
# KL Divergence Calculation
# =============================================================================

@dataclass
class LocalEnvironment:
    """Represents a local environment descriptor."""
    center_sublattice: str  # "Ni1", "Ni2", or "Si1"
    n_ni_neighbors: int
    n_si_neighbors: int
    
    def to_key(self) -> str:
        """Convert to hashable key."""
        return f"{self.center_sublattice}:{self.n_ni_neighbors}:{self.n_si_neighbors}"


def calculate_local_environments(
    structure: CrystalStructure,
    config: Configuration,
    neighbor_list: Dict[int, List[Tuple[int, float]]],
) -> List[LocalEnvironment]:
    """
    Calculate local environment descriptors for all sites.
    
    Args:
        structure: CrystalStructure object
        config: Configuration object with occupancy
        neighbor_list: Pre-calculated neighbor list
    
    Returns:
        List of LocalEnvironment objects (one per site)
    """
    environments = []
    
    for site in structure.sites:
        # Count neighbors by species
        n_ni = 0
        n_si = 0
        
        for neighbor_idx, _ in neighbor_list[site.index]:
            if config.occupancy[neighbor_idx] == "Ni":
                n_ni += 1
            else:
                n_si += 1
        
        env = LocalEnvironment(
            center_sublattice=site.sublattice,
            n_ni_neighbors=n_ni,
            n_si_neighbors=n_si,
        )
        environments.append(env)
    
    return environments


def calculate_environment_distribution(
    environments: List[LocalEnvironment],
) -> Dict[str, float]:
    """
    Calculate probability distribution of local environments.
    
    Args:
        environments: List of LocalEnvironment objects
    
    Returns:
        Dictionary mapping environment key to probability
    """
    counts = defaultdict(int)
    total = len(environments)
    
    for env in environments:
        key = env.to_key()
        counts[key] += 1
    
    return {key: count / total for key, count in counts.items()}


def calculate_kl_divergence(
    P: Dict[str, float],
    Q: Dict[str, float],
    smoothing: float = 1e-10,
) -> float:
    """
    Calculate KL divergence D_KL(P || Q).
    
    D_KL = Σ P(σ) ln(P(σ) / Q(σ))
    
    Uses Laplace smoothing to handle zero frequencies.
    
    Args:
        P: Target distribution
        Q: Reference distribution (baseline)
        smoothing: Smoothing parameter for zero frequencies
    
    Returns:
        KL divergence value
    """
    # Get all keys from both distributions
    all_keys = set(P.keys()) | set(Q.keys())
    
    kl_div = 0.0
    
    for key in all_keys:
        p = P.get(key, 0.0)
        q = max(Q.get(key, 0.0), smoothing)  # Apply smoothing to Q
        
        if p > 0:
            kl_div += p * np.log(p / q)
    
    return kl_div


def calculate_baseline_distribution(
    structure: CrystalStructure,
    group_a_configs: List[Configuration],
    neighbor_list: Dict[int, List[Tuple[int, float]]],
) -> Dict[str, float]:
    """
    Calculate baseline distribution Q(σ) from Group A random structures.
    
    Args:
        structure: CrystalStructure object
        group_a_configs: List of Group A configurations
        neighbor_list: Pre-calculated neighbor list
    
    Returns:
        Baseline distribution Q(σ)
    """
    all_counts = defaultdict(int)
    total = 0
    
    for config in group_a_configs:
        environments = calculate_local_environments(structure, config, neighbor_list)
        for env in environments:
            key = env.to_key()
            all_counts[key] += 1
            total += 1
    
    return {key: count / total for key, count in all_counts.items()}


def analyze_kl_divergence(
    structure: CrystalStructure,
    configurations: List[Configuration],
    baseline_Q: Dict[str, float],
    neighbor_list: Dict[int, List[Tuple[int, float]]],
) -> List[Dict]:
    """
    Calculate KL divergence for all configurations.
    
    Args:
        structure: CrystalStructure object
        configurations: List of Configuration objects
        baseline_Q: Baseline distribution from Group A
        neighbor_list: Pre-calculated neighbor list
    
    Returns:
        List of analysis results (one dict per configuration)
    """
    results = []
    
    for config in configurations:
        environments = calculate_local_environments(structure, config, neighbor_list)
        P = calculate_environment_distribution(environments)
        kl_value = calculate_kl_divergence(P, baseline_Q)
        
        results.append({
            "structure_id": config.structure_id,
            "group": config.group,
            "composition": config.composition,
            "n_ni": config.n_ni,
            "n_si": config.n_si,
            "kl_divergence": kl_value,
            "metadata": config.metadata,
        })
    
    return results


# =============================================================================
# Large-Scale Candidate Generation and Selection
# =============================================================================

def generate_large_candidate_pool(
    structure: CrystalStructure,
    n_candidates_per_composition: int = 5000,
    pair_potential_k: float = DEFAULT_PAIR_POTENTIAL_K,
    verbose: bool = True,
) -> Tuple[List[Configuration], Dict[str, List[Configuration]]]:
    """
    Generate a large pool of candidate structures using multiple strategies.
    
    Strategies used:
    1. Random configurations (for baseline and diversity)
    2. SA-optimized configurations (pair potential minimization)
    3. KL-maximization SA (direct ordering optimization)
    4. Site-specific substitution patterns
    
    Args:
        structure: CrystalStructure object
        n_candidates_per_composition: Number of candidates per composition
        pair_potential_k: Exponent for pair potential
        verbose: Print progress messages
    
    Returns:
        Tuple of (all_candidates, candidates_by_composition)
    """
    # Composition specifications: (n_ni, n_si)
    compositions = [
        (32, 16),  # Stoichiometric
        (34, 14),  # Ni-rich
        (30, 18),  # Si-rich
    ]
    
    candidates_by_composition: Dict[str, List[Configuration]] = {}
    all_candidates: List[Configuration] = []
    seen_hashes: Set[str] = set()
    
    # Build pair list for SA
    pair_list = build_pair_list_with_weights(structure, SA_CUTOFF, pair_potential_k)
    
    for n_ni, n_si in compositions:
        comp_key = f"Ni{n_ni}Si{n_si}"
        candidates_by_composition[comp_key] = []
        
        if verbose:
            print(f"  Generating candidates for {comp_key}...")
        
        config_idx = 0
        
        # Strategy 1: Random configurations (40% of candidates)
        n_random = int(n_candidates_per_composition * 0.4)
        for i in range(n_random):
            seed = hash((comp_key, "random", i)) % (2**31)
            config = generate_random_configuration(structure, n_ni, n_si, seed)
            config_hash = config.get_occupancy_hash()
            
            if config_hash not in seen_hashes:
                seen_hashes.add(config_hash)
                config.structure_id = f"CAND_{comp_key}_{config_idx:05d}"
                config.group = "candidate"
                config.metadata["strategy"] = "random"
                candidates_by_composition[comp_key].append(config)
                all_candidates.append(config)
                config_idx += 1
        
        if verbose:
            print(f"    - Random: {config_idx} unique structures")
        
        # Strategy 2: SA-optimized (pair potential minimization) (30% of candidates)
        n_sa = int(n_candidates_per_composition * 0.3)
        sa_collected = 0
        sa_run = 0
        
        while sa_collected < n_sa and sa_run < n_sa * 5:
            seed = hash((comp_key, "sa", sa_run)) % (2**31)
            
            # Vary SA parameters for diversity
            T_initial = 5.0 + (sa_run % 10) * 2.0
            cooling_rate = 0.90 + (sa_run % 5) * 0.02
            
            final_config, trajectory = simulated_annealing(
                structure=structure,
                n_ni=n_ni,
                n_si=n_si,
                pair_list=pair_list,
                T_initial=T_initial,
                cooling_rate=cooling_rate,
                random_seed=seed,
                collect_trajectory=True,
            )
            
            # Sample from trajectory
            if trajectory:
                # Sample at different points in trajectory
                sample_indices = [
                    len(trajectory) // 4,
                    len(trajectory) // 2,
                    3 * len(trajectory) // 4,
                    len(trajectory) - 1,
                ]
                
                for idx in sample_indices:
                    if sa_collected >= n_sa:
                        break
                    if idx < len(trajectory):
                        _, config = trajectory[idx]
                        config_hash = config.get_occupancy_hash()
                        
                        if config_hash not in seen_hashes:
                            seen_hashes.add(config_hash)
                            config.structure_id = f"CAND_{comp_key}_{config_idx:05d}"
                            config.group = "candidate"
                            config.metadata["strategy"] = "sa_pair_potential"
                            config.metadata["sa_T_initial"] = T_initial
                            candidates_by_composition[comp_key].append(config)
                            all_candidates.append(config)
                            config_idx += 1
                            sa_collected += 1
            
            sa_run += 1
        
        if verbose:
            print(f"    - SA (pair potential): {sa_collected} unique structures")
        
        # Strategy 3: Site-specific substitutions (20% of candidates)
        n_site_specific = int(n_candidates_per_composition * 0.2)
        site_collected = 0
        
        # Calculate how many substitutions needed for this composition
        # Stoichiometric is 32 Ni, 16 Si
        # For Ni34Si14: need 2 more Ni (Si->Ni substitutions on Si1)
        # For Ni30Si18: need 2 more Si (Ni->Si substitutions on Ni1 or Ni2)
        
        sublattice_indices = structure.get_sublattice_indices()
        
        if n_ni == 32 and n_si == 16:
            # Stoichiometric - create antisite defect pairs
            for trial in range(n_site_specific):
                seed = hash((comp_key, "site_specific", trial)) % (2**31)
                random.seed(seed)
                
                # Random antisite: swap one Ni with one Si
                occupancy = [site.ideal_species for site in structure.sites]
                
                # Choose random Ni site and Si site to swap
                ni_sites = sublattice_indices["Ni1"] + sublattice_indices["Ni2"]
                si_sites = sublattice_indices["Si1"]
                
                ni_to_swap = random.choice(ni_sites)
                si_to_swap = random.choice(si_sites)
                
                occupancy[ni_to_swap] = "Si"
                occupancy[si_to_swap] = "Ni"
                
                config = Configuration(
                    structure_id=f"CAND_{comp_key}_{config_idx:05d}",
                    group="candidate",
                    composition=comp_key,
                    occupancy=occupancy,
                    n_ni=n_ni,
                    n_si=n_si,
                    metadata={"strategy": "antisite_pair", "trial": trial},
                )
                
                config_hash = config.get_occupancy_hash()
                if config_hash not in seen_hashes:
                    seen_hashes.add(config_hash)
                    candidates_by_composition[comp_key].append(config)
                    all_candidates.append(config)
                    config_idx += 1
                    site_collected += 1
        
        elif n_ni > 32:
            # Ni-rich: Si->Ni substitutions
            n_excess_ni = n_ni - 32
            for trial in range(n_site_specific):
                seed = hash((comp_key, "site_specific", trial)) % (2**31)
                random.seed(seed)
                
                occupancy = [site.ideal_species for site in structure.sites]
                si_sites = sublattice_indices["Si1"].copy()
                random.shuffle(si_sites)
                
                for i in range(min(n_excess_ni, len(si_sites))):
                    occupancy[si_sites[i]] = "Ni"
                
                config = Configuration(
                    structure_id=f"CAND_{comp_key}_{config_idx:05d}",
                    group="candidate",
                    composition=comp_key,
                    occupancy=occupancy,
                    n_ni=sum(1 for s in occupancy if s == "Ni"),
                    n_si=sum(1 for s in occupancy if s == "Si"),
                    metadata={"strategy": "si_to_ni_substitution", "trial": trial},
                )
                
                config_hash = config.get_occupancy_hash()
                if config_hash not in seen_hashes:
                    seen_hashes.add(config_hash)
                    candidates_by_composition[comp_key].append(config)
                    all_candidates.append(config)
                    config_idx += 1
                    site_collected += 1
        
        else:
            # Si-rich: Ni->Si substitutions
            n_excess_si = n_si - 16
            for trial in range(n_site_specific):
                seed = hash((comp_key, "site_specific", trial)) % (2**31)
                random.seed(seed)
                
                occupancy = [site.ideal_species for site in structure.sites]
                
                # Randomly choose Ni1 or Ni2 sublattice
                if random.random() < 0.5:
                    ni_sites = sublattice_indices["Ni1"].copy()
                else:
                    ni_sites = sublattice_indices["Ni2"].copy()
                random.shuffle(ni_sites)
                
                for i in range(min(n_excess_si, len(ni_sites))):
                    occupancy[ni_sites[i]] = "Si"
                
                config = Configuration(
                    structure_id=f"CAND_{comp_key}_{config_idx:05d}",
                    group="candidate",
                    composition=comp_key,
                    occupancy=occupancy,
                    n_ni=sum(1 for s in occupancy if s == "Ni"),
                    n_si=sum(1 for s in occupancy if s == "Si"),
                    metadata={"strategy": "ni_to_si_substitution", "trial": trial},
                )
                
                config_hash = config.get_occupancy_hash()
                if config_hash not in seen_hashes:
                    seen_hashes.add(config_hash)
                    candidates_by_composition[comp_key].append(config)
                    all_candidates.append(config)
                    config_idx += 1
                    site_collected += 1
        
        if verbose:
            print(f"    - Site-specific: {site_collected} unique structures")
        
        # Strategy 4: Additional random to fill remaining (10%)
        n_remaining = n_candidates_per_composition - len(candidates_by_composition[comp_key])
        for i in range(n_remaining):
            seed = hash((comp_key, "fill", i)) % (2**31)
            config = generate_random_configuration(structure, n_ni, n_si, seed)
            config_hash = config.get_occupancy_hash()
            
            if config_hash not in seen_hashes:
                seen_hashes.add(config_hash)
                config.structure_id = f"CAND_{comp_key}_{config_idx:05d}"
                config.group = "candidate"
                config.metadata["strategy"] = "random_fill"
                candidates_by_composition[comp_key].append(config)
                all_candidates.append(config)
                config_idx += 1
        
        if verbose:
            print(f"    - Total for {comp_key}: {len(candidates_by_composition[comp_key])} structures")
    
    return all_candidates, candidates_by_composition


def calculate_baseline_distribution_by_composition(
    structure: CrystalStructure,
    candidates_by_composition: Dict[str, List[Configuration]],
    neighbor_list: Dict[int, List[Tuple[int, float]]],
    n_baseline_samples: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate baseline distribution Q(σ) separately for each composition.
    
    Uses random configurations from the candidate pool to estimate Q(σ).
    
    Args:
        structure: CrystalStructure object
        candidates_by_composition: Candidates grouped by composition
        neighbor_list: Pre-calculated neighbor list
        n_baseline_samples: Number of random samples for baseline estimation
    
    Returns:
        Dictionary mapping composition to baseline distribution Q(σ)
    """
    baseline_by_composition: Dict[str, Dict[str, float]] = {}
    
    for comp_key, candidates in candidates_by_composition.items():
        # Use random strategy candidates for baseline
        random_candidates = [
            c for c in candidates 
            if c.metadata.get("strategy") in ["random", "random_fill"]
        ]
        
        # If not enough random candidates, use all candidates
        if len(random_candidates) < n_baseline_samples:
            random_candidates = candidates
        
        # Sample for baseline
        sample_size = min(n_baseline_samples, len(random_candidates))
        baseline_sample = random_candidates[:sample_size]
        
        # Calculate baseline distribution
        all_counts = defaultdict(int)
        total = 0
        
        for config in baseline_sample:
            environments = calculate_local_environments(structure, config, neighbor_list)
            for env in environments:
                key = env.to_key()
                all_counts[key] += 1
                total += 1
        
        baseline_by_composition[comp_key] = {
            key: count / total for key, count in all_counts.items()
        }
    
    return baseline_by_composition


def rank_and_select_top_structures(
    structure: CrystalStructure,
    candidates_by_composition: Dict[str, List[Configuration]],
    baseline_by_composition: Dict[str, Dict[str, float]],
    neighbor_list: Dict[int, List[Tuple[int, float]]],
    n_select: int = 256,
    composition_ratio: Optional[Dict[str, int]] = None,
    verbose: bool = True,
) -> List[Configuration]:
    """
    Rank all candidates by KL divergence and select top structures.
    
    Args:
        structure: CrystalStructure object
        candidates_by_composition: Candidates grouped by composition
        baseline_by_composition: Baseline Q(σ) for each composition
        neighbor_list: Pre-calculated neighbor list
        n_select: Total number of structures to select
        composition_ratio: Optional dict specifying how many to select per composition
                          If None, selects proportionally or by pure KL ranking
        verbose: Print progress messages
    
    Returns:
        List of selected Configuration objects (sorted by KL divergence descending)
    """
    # Calculate KL divergence for all candidates
    all_results: List[Tuple[float, Configuration]] = []
    
    for comp_key, candidates in candidates_by_composition.items():
        baseline_Q = baseline_by_composition[comp_key]
        
        if verbose:
            print(f"  Calculating KL divergence for {comp_key} ({len(candidates)} candidates)...")
        
        for config in candidates:
            environments = calculate_local_environments(structure, config, neighbor_list)
            P = calculate_environment_distribution(environments)
            kl_value = calculate_kl_divergence(P, baseline_Q)
            
            config.metadata["kl_divergence"] = kl_value
            all_results.append((kl_value, config))
    
    # Sort by KL divergence (descending - highest first)
    all_results.sort(key=lambda x: x[0], reverse=True)
    
    if verbose:
        print(f"  KL divergence range: {all_results[-1][0]:.4f} - {all_results[0][0]:.4f}")
    
    # Select top structures
    if composition_ratio is not None:
        # Select specified number per composition
        selected: List[Configuration] = []
        results_by_comp: Dict[str, List[Tuple[float, Configuration]]] = defaultdict(list)
        
        for kl, config in all_results:
            results_by_comp[config.composition].append((kl, config))
        
        for comp_key, n_select_comp in composition_ratio.items():
            comp_results = results_by_comp.get(comp_key, [])
            for i, (kl, config) in enumerate(comp_results[:n_select_comp]):
                config.structure_id = f"SEL_{i:03d}_{comp_key}"
                config.group = "selected"
                selected.append(config)
        
        # Sort selected by KL divergence
        selected.sort(key=lambda c: c.metadata.get("kl_divergence", 0), reverse=True)
    else:
        # Select top n_select overall
        selected = []
        for i, (kl, config) in enumerate(all_results[:n_select]):
            config.structure_id = f"SEL_{i:03d}"
            config.group = "selected"
            selected.append(config)
    
    if verbose:
        # Report composition distribution
        comp_counts = defaultdict(int)
        for config in selected:
            comp_counts[config.composition] += 1
        print(f"  Selected {len(selected)} structures:")
        for comp, count in sorted(comp_counts.items()):
            print(f"    - {comp}: {count}")
    
    return selected


def run_full_workflow_with_selection(
    output_dir: str = "project_Ni2Si",
    n_candidates_per_composition: int = 5000,
    n_select: int = 256,
    pair_potential_k: float = DEFAULT_PAIR_POTENTIAL_K,
    composition_ratio: Optional[Dict[str, int]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run the complete workflow with large-scale candidate generation and selection.
    
    This function:
    1. Creates the project directory structure
    2. Generates the crystal structure and site mapping
    3. Generates a large candidate pool (thousands of structures)
    4. Calculates composition-specific baseline Q(σ)
    5. Ranks all candidates by KL divergence
    6. Selects top 256 structures with highest ordering
    7. Creates VASP input files for selected structures
    
    Args:
        output_dir: Output directory path
        n_candidates_per_composition: Number of candidates to generate per composition
        n_select: Number of structures to select (default: 256)
        pair_potential_k: Exponent for pair potential (default: 1)
        composition_ratio: Optional dict specifying selection per composition
        verbose: Print progress messages
    
    Returns:
        Dictionary with workflow results
    """
    if verbose:
        print("=" * 70)
        print("δ-Ni₂Si Antisite Defect Formation Energy Analysis")
        print("Large-Scale Candidate Generation and Selection Workflow")
        print("=" * 70)
    
    # Create directory structure
    if verbose:
        print("\n[1/7] Creating project directory structure...")
    dirs = create_project_directory(output_dir)
    
    # Generate crystal structure
    if verbose:
        print("[2/7] Generating crystal structure and site mapping...")
    structure = generate_ni2si_supercell()
    
    # Save site mapping
    site_mapping_path = os.path.join(dirs["structures"], "site_mapping.json")
    save_site_mapping(structure, site_mapping_path)
    if verbose:
        print(f"  - Site mapping saved to: {site_mapping_path}")
        print(f"  - Total atoms: {len(structure.sites)}")
    
    # Build neighbor list
    if verbose:
        print("[3/7] Building neighbor list...")
    neighbor_list = build_neighbor_list(structure, KL_CUTOFF)
    
    # Generate large candidate pool
    if verbose:
        print(f"[4/7] Generating large candidate pool ({n_candidates_per_composition} per composition)...")
    all_candidates, candidates_by_composition = generate_large_candidate_pool(
        structure=structure,
        n_candidates_per_composition=n_candidates_per_composition,
        pair_potential_k=pair_potential_k,
        verbose=verbose,
    )
    
    if verbose:
        print(f"  - Total candidates generated: {len(all_candidates)}")
    
    # Calculate composition-specific baseline distributions
    if verbose:
        print("[5/7] Calculating composition-specific baseline distributions...")
    baseline_by_composition = calculate_baseline_distribution_by_composition(
        structure=structure,
        candidates_by_composition=candidates_by_composition,
        neighbor_list=neighbor_list,
    )
    
    # Rank and select top structures
    if verbose:
        print(f"[6/7] Ranking candidates and selecting top {n_select} structures...")
    selected_configs = rank_and_select_top_structures(
        structure=structure,
        candidates_by_composition=candidates_by_composition,
        baseline_by_composition=baseline_by_composition,
        neighbor_list=neighbor_list,
        n_select=n_select,
        composition_ratio=composition_ratio,
        verbose=verbose,
    )
    
    # Generate VASP input files for selected structures
    if verbose:
        print("[7/7] Generating VASP input files for selected structures...")
    
    # Save INCAR template
    incar_template = generate_incar(selected_configs[0])
    incar_path = os.path.join(dirs["vasp_runs"], "INCAR_template")
    with open(incar_path, 'w') as f:
        f.write(incar_template)
    
    # Save KPOINTS
    kpoints_content = generate_kpoints()
    kpoints_path = os.path.join(dirs["vasp_runs"], "KPOINTS")
    with open(kpoints_path, 'w') as f:
        f.write(kpoints_content)
    
    # Save POSCAR files for selected structures
    for config in selected_configs:
        poscar_content = generate_poscar(structure, config)
        poscar_path = os.path.join(dirs["structures"], f"POSCAR_{config.structure_id}")
        with open(poscar_path, 'w') as f:
            f.write(poscar_content)
    
    if verbose:
        print(f"  - POSCAR files saved to: {dirs['structures']}")
        print(f"  - INCAR template saved to: {incar_path}")
        print(f"  - KPOINTS saved to: {kpoints_path}")
    
    # Save results to CSV
    import csv
    csv_path = os.path.join(dirs["analysis"], "energy_kl.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "structure_id", "composition", "n_ni", "n_si", 
            "kl_divergence", "strategy", "energy", "mag_mom"
        ])
        writer.writeheader()
        for config in selected_configs:
            writer.writerow({
                "structure_id": config.structure_id,
                "composition": config.composition,
                "n_ni": config.n_ni,
                "n_si": config.n_si,
                "kl_divergence": config.metadata.get("kl_divergence", ""),
                "strategy": config.metadata.get("strategy", ""),
                "energy": "",  # To be filled after VASP calculations
                "mag_mom": "",  # To be filled after VASP calculations
            })
    
    # Save candidate pool statistics
    stats_path = os.path.join(dirs["analysis"], "candidate_pool_stats.json")
    stats = {
        "total_candidates": len(all_candidates),
        "candidates_per_composition": {
            comp: len(configs) for comp, configs in candidates_by_composition.items()
        },
        "selected_count": len(selected_configs),
        "kl_range": {
            "min": min(c.metadata.get("kl_divergence", 0) for c in selected_configs),
            "max": max(c.metadata.get("kl_divergence", 0) for c in selected_configs),
        },
        "composition_distribution": {
            comp: sum(1 for c in selected_configs if c.composition == comp)
            for comp in candidates_by_composition.keys()
        },
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    if verbose:
        print(f"  - Results saved to: {csv_path}")
        print(f"  - Statistics saved to: {stats_path}")
        print("\n" + "=" * 70)
        print("Workflow completed successfully!")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  - Generated {len(all_candidates)} candidate structures")
        print(f"  - Selected top {len(selected_configs)} by KL divergence (highest ordering)")
        print(f"\nNext steps:")
        print(f"1. Copy POTCAR files to {dirs['vasp_runs']}")
        print(f"2. Submit VASP jobs for {len(selected_configs)} selected structures")
        print(f"3. Update {csv_path} with calculated energies")
        print(f"4. Run LOBSTER for COHP/COOP analysis")
    
    return {
        "structure": structure,
        "all_candidates": all_candidates,
        "candidates_by_composition": candidates_by_composition,
        "selected_configurations": selected_configs,
        "baseline_by_composition": baseline_by_composition,
        "directories": dirs,
        "statistics": stats,
    }


# =============================================================================
# Main Workflow Functions (Legacy - kept for backward compatibility)
# =============================================================================

def create_project_directory(base_path: str) -> Dict[str, str]:
    """
    Create the project directory structure.
    
    Args:
        base_path: Base path for the project
    
    Returns:
        Dictionary of directory paths
    """
    dirs = {
        "base": base_path,
        "structures": os.path.join(base_path, "01_structures"),
        "vasp_runs": os.path.join(base_path, "02_vasp_runs"),
        "analysis": os.path.join(base_path, "03_analysis"),
        "scripts": os.path.join(base_path, "04_scripts"),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(dirs["analysis"], "dos_plots"), exist_ok=True)
    
    return dirs


def run_full_workflow(
    output_dir: str = "project_Ni2Si",
    pair_potential_k: float = DEFAULT_PAIR_POTENTIAL_K,
    verbose: bool = True,
) -> Dict:
    """
    Run the complete workflow for δ-Ni₂Si antisite defect analysis.
    
    This function:
    1. Creates the project directory structure
    2. Generates the crystal structure and site mapping
    3. Generates all 256 configurations (Groups A, B, C)
    4. Creates VASP input files
    5. Calculates KL divergence for all structures
    
    Args:
        output_dir: Output directory path
        pair_potential_k: Exponent for pair potential (default: 1)
        verbose: Print progress messages
    
    Returns:
        Dictionary with workflow results
    """
    if verbose:
        print("=" * 60)
        print("δ-Ni₂Si Antisite Defect Formation Energy Analysis")
        print("=" * 60)
    
    # Create directory structure
    if verbose:
        print("\n[1/6] Creating project directory structure...")
    dirs = create_project_directory(output_dir)
    
    # Generate crystal structure
    if verbose:
        print("[2/6] Generating crystal structure and site mapping...")
    structure = generate_ni2si_supercell()
    
    # Save site mapping
    site_mapping_path = os.path.join(dirs["structures"], "site_mapping.json")
    save_site_mapping(structure, site_mapping_path)
    if verbose:
        print(f"  - Site mapping saved to: {site_mapping_path}")
        print(f"  - Total atoms: {len(structure.sites)}")
        sublattice_indices = structure.get_sublattice_indices()
        for sublattice, indices in sublattice_indices.items():
            print(f"    - {sublattice}: {len(indices)} sites")
    
    # Build neighbor lists
    if verbose:
        print("[3/6] Building neighbor lists...")
    neighbor_list_kl = build_neighbor_list(structure, KL_CUTOFF)
    
    # Generate Group A structures
    if verbose:
        print("[4/6] Generating structure configurations...")
        print("  - Group A (Random baseline): 100 structures")
    group_a = generate_group_a_structures(structure)
    
    if verbose:
        print("  - Group B (SA-optimized): 100 structures")
    group_b = generate_group_b_structures(structure, k=pair_potential_k)
    
    if verbose:
        print("  - Group C (Specific defects): 56 structures")
    group_c = generate_group_c_structures(structure)
    
    all_configs = group_a + group_b + group_c
    if verbose:
        print(f"  - Total configurations: {len(all_configs)}")
    
    # Generate VASP input files
    if verbose:
        print("[5/6] Generating VASP input files...")
    
    # Save INCAR template
    incar_template = generate_incar(all_configs[0])
    incar_path = os.path.join(dirs["vasp_runs"], "INCAR_template")
    with open(incar_path, 'w') as f:
        f.write(incar_template)
    
    # Save KPOINTS
    kpoints_content = generate_kpoints()
    kpoints_path = os.path.join(dirs["vasp_runs"], "KPOINTS")
    with open(kpoints_path, 'w') as f:
        f.write(kpoints_content)
    
    # Save POSCAR files
    for config in all_configs:
        poscar_content = generate_poscar(structure, config)
        poscar_path = os.path.join(dirs["structures"], f"POSCAR_{config.structure_id}")
        with open(poscar_path, 'w') as f:
            f.write(poscar_content)
    
    if verbose:
        print(f"  - POSCAR files saved to: {dirs['structures']}")
        print(f"  - INCAR template saved to: {incar_path}")
        print(f"  - KPOINTS saved to: {kpoints_path}")
    
    # Calculate KL divergence
    if verbose:
        print("[6/6] Calculating KL divergence...")
    
    # Calculate baseline distribution from Group A
    baseline_Q = calculate_baseline_distribution(structure, group_a, neighbor_list_kl)
    
    # Analyze all configurations
    kl_results = analyze_kl_divergence(structure, all_configs, baseline_Q, neighbor_list_kl)
    
    # Save results to CSV
    import csv
    csv_path = os.path.join(dirs["analysis"], "energy_kl.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "structure_id", "group", "composition", "n_ni", "n_si", 
            "kl_divergence", "energy", "mag_mom"
        ])
        writer.writeheader()
        for result in kl_results:
            writer.writerow({
                "structure_id": result["structure_id"],
                "group": result["group"],
                "composition": result["composition"],
                "n_ni": result["n_ni"],
                "n_si": result["n_si"],
                "kl_divergence": result["kl_divergence"],
                "energy": "",  # To be filled after VASP calculations
                "mag_mom": "",  # To be filled after VASP calculations
            })
    
    if verbose:
        print(f"  - KL divergence results saved to: {csv_path}")
        print("\n" + "=" * 60)
        print("Workflow completed successfully!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Copy POTCAR files to {dirs['vasp_runs']}")
        print(f"2. Submit VASP jobs for all structures")
        print(f"3. Update {csv_path} with calculated energies")
        print(f"4. Run LOBSTER for COHP/COOP analysis")
    
    return {
        "structure": structure,
        "configurations": all_configs,
        "kl_results": kl_results,
        "baseline_Q": baseline_Q,
        "directories": dirs,
    }


# =============================================================================
# Streamlit Application (Optional)
# =============================================================================

def create_streamlit_app():
    """
    Create a Streamlit application for interactive analysis.
    
    This function is called when running the module directly with Streamlit.
    """
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    
    st.set_page_config(
        page_title="δ-Ni₂Si Antisite Defect Analysis",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 δ-Ni₂Si Antisite Defect Formation Energy Analysis")
    st.markdown("""
    This application implements a comprehensive workflow for analyzing antisite defects
    in δ-Ni₂Si (Pnma structure) using KL divergence for short-range order (SRO) analysis.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    pair_potential_k = st.sidebar.slider(
        "Pair Potential Exponent (k)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Exponent in H = Σ ε_αβ/r_ij^k"
    )
    
    output_dir = st.sidebar.text_input(
        "Output Directory",
        value="project_Ni2Si",
        help="Directory for output files"
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Structure Generation",
        "🔍 KL Divergence Analysis",
        "📁 VASP Files",
        "📚 Documentation"
    ])
    
    with tab1:
        st.header("Crystal Structure and Configuration Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Crystal Structure Parameters")
            st.markdown(f"""
            - **Space Group**: Pnma (#62)
            - **Lattice Constants**: a={LATTICE_A}Å, b={LATTICE_B}Å, c={LATTICE_C}Å
            - **Supercell**: {SUPERCELL_X}×{SUPERCELL_Y}×{SUPERCELL_Z}
            - **Total Atoms**: 48 (stoichiometric: 32 Ni + 16 Si)
            """)
            
            st.subheader("Wyckoff 4c Positions")
            for sublattice, coords in WYCKOFF_4C.items():
                st.markdown(f"- **{sublattice}**: ({coords[0]}, {coords[1]}, {coords[2]})")
        
        with col2:
            st.subheader("Configuration Groups")
            st.markdown("""
            **Group A (Random Baseline)**: 100 structures
            - Ni₃₂Si₁₆: 40 structures
            - Ni₃₄Si₁₄: 30 structures
            - Ni₃₀Si₁₈: 30 structures
            
            **Group B (SA-Optimized)**: 100 structures
            - Sampled from simulated annealing trajectories
            
            **Group C (Specific Defects)**: 56 structures
            - Site-specific substitutions for DVM comparison
            """)
        
        if st.button("Generate All Structures", type="primary"):
            with st.spinner("Generating structures..."):
                results = run_full_workflow(
                    output_dir=output_dir,
                    pair_potential_k=pair_potential_k,
                    verbose=False
                )
                st.session_state["results"] = results
            
            st.success(f"Generated {len(results['configurations'])} configurations!")
            st.info(f"Files saved to: {output_dir}/")
    
    with tab2:
        st.header("KL Divergence Analysis")
        
        if "results" not in st.session_state:
            st.warning("Please generate structures first in the 'Structure Generation' tab.")
        else:
            results = st.session_state["results"]
            kl_results = results["kl_results"]
            
            # Create DataFrame for visualization
            import pandas as pd
            df = pd.DataFrame(kl_results)
            
            # KL divergence distribution
            st.subheader("KL Divergence Distribution by Group")
            
            fig = px.histogram(
                df, x="kl_divergence", color="group",
                barmode="overlay", opacity=0.7,
                labels={"kl_divergence": "KL Divergence", "group": "Group"},
                title="Distribution of KL Divergence Values"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            summary = df.groupby("group")["kl_divergence"].agg(["mean", "std", "min", "max"])
            st.dataframe(summary)
            
            # Composition vs KL divergence
            st.subheader("Composition vs KL Divergence")
            fig2 = px.scatter(
                df, x="n_ni", y="kl_divergence", color="group",
                labels={"n_ni": "Number of Ni Atoms", "kl_divergence": "KL Divergence"},
                title="KL Divergence vs Composition"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.header("VASP Input Files")
        
        st.markdown("""
        ### INCAR Template
        The following INCAR settings are used for all calculations:
        """)
        
        st.code(generate_incar(Configuration(
            structure_id="example",
            group="A",
            composition="Ni32Si16",
            occupancy=["Ni"] * 32 + ["Si"] * 16,
            n_ni=32,
            n_si=16
        )), language="ini")
        
        st.markdown("""
        ### KPOINTS
        """)
        st.code(generate_kpoints(), language="text")
        
        st.markdown("""
        ### Directory Structure
        ```
        project_Ni2Si/
        ├── 01_structures/        # Generated POSCAR files
        │   ├── site_mapping.json
        │   ├── POSCAR_A_000
        │   ├── POSCAR_A_001
        │   └── ...
        ├── 02_vasp_runs/         # VASP calculation directories
        │   ├── INCAR_template
        │   └── KPOINTS
        ├── 03_analysis/
        │   ├── energy_kl.csv
        │   └── dos_plots/
        └── 04_scripts/
        ```
        """)
    
    with tab4:
        st.header("Documentation")
        
        st.markdown("""
        ## Theory Background
        
        ### δ-Ni₂Si Crystal Structure
        
        δ-Ni₂Si crystallizes in the orthorhombic Pnma space group (#62) with:
        - 12 atoms per unit cell (8 Ni + 4 Si)
        - Three distinct Wyckoff 4c sites: Ni1, Ni2, and Si1
        
        ### KL Divergence for SRO Analysis
        
        The Kullback-Leibler divergence measures the difference between two probability
        distributions. For SRO analysis:
        
        $$D_{KL}(P || Q) = \\sum_{\\sigma} P(\\sigma) \\ln\\frac{P(\\sigma)}{Q(\\sigma)}$$
        
        where:
        - $P(\\sigma)$: Distribution of local environments in the target structure
        - $Q(\\sigma)$: Baseline distribution from random structures
        - $\\sigma$: Local environment descriptor (center site type + neighbor counts)
        
        ### Simulated Annealing
        
        The pair potential Hamiltonian used for SA:
        
        $$H = \\sum_{i<j} \\frac{\\epsilon_{\\alpha\\beta}}{r_{ij}^k}$$
        
        with parameters:
        - $\\epsilon_{Ni-Si} = -1.0$ (attractive)
        - $\\epsilon_{Ni-Ni} = 0.5$ (repulsive)
        - $\\epsilon_{Si-Si} = 0.8$ (repulsive)
        
        ### References
        
        1. Warren-Cowley SRO parameters
        2. DVM (Discrete Variational Method) for electronic structure
        3. LOBSTER for COHP/COOP analysis
        """)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        create_streamlit_app()
    else:
        # Run the workflow directly
        results = run_full_workflow(verbose=True)
