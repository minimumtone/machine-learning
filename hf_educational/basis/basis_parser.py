"""
Basis set parser and database.

Contains basis set definitions for common basis sets (STO-3G, 6-31G, etc.)
in a format compatible with Gaussian basis set specifications.
"""

import json
from typing import Dict, List, Any


STO_3G_BASIS = {
    1: [  # Hydrogen
        {
            'shell_type': 's',
            'exponents': [3.42525091, 0.62391373, 0.16885540],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        }
    ],
    2: [  # Helium
        {
            'shell_type': 's',
            'exponents': [6.36242139, 1.15892300, 0.31364979],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        }
    ],
    3: [  # Lithium
        {
            'shell_type': 's',
            'exponents': [16.11957475, 2.93620070, 0.79465050],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        },
        {
            'shell_type': 's',
            'exponents': [0.63628970, 0.14786010, 0.04808870],
            'coefficients': [-0.09996723, 0.39951283, 0.70011547]
        },
        {
            'shell_type': 'p',
            'exponents': [0.63628970, 0.14786010, 0.04808870],
            'coefficients': [0.15591627, 0.60768372, 0.39195739]
        }
    ],
    4: [  # Beryllium
        {
            'shell_type': 's',
            'exponents': [30.16787069, 5.49513122, 1.48735132],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        },
        {
            'shell_type': 's',
            'exponents': [1.31138778, 0.30552053, 0.09937298],
            'coefficients': [-0.09996723, 0.39951283, 0.70011547]
        },
        {
            'shell_type': 'p',
            'exponents': [1.31138778, 0.30552053, 0.09937298],
            'coefficients': [0.15591627, 0.60768372, 0.39195739]
        }
    ],
    5: [  # Boron
        {
            'shell_type': 's',
            'exponents': [48.79134554, 8.88706748, 2.40572695],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        },
        {
            'shell_type': 's',
            'exponents': [2.23679478, 0.52140510, 0.16955180],
            'coefficients': [-0.09996723, 0.39951283, 0.70011547]
        },
        {
            'shell_type': 'p',
            'exponents': [2.23679478, 0.52140510, 0.16955180],
            'coefficients': [0.15591627, 0.60768372, 0.39195739]
        }
    ],
    6: [  # Carbon
        {
            'shell_type': 's',
            'exponents': [71.6168370, 13.0450960, 3.5305122],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        },
        {
            'shell_type': 's',
            'exponents': [2.9412494, 0.6834831, 0.2222899],
            'coefficients': [-0.09996723, 0.39951283, 0.70011547]
        },
        {
            'shell_type': 'p',
            'exponents': [2.9412494, 0.6834831, 0.2222899],
            'coefficients': [0.15591627, 0.60768372, 0.39195739]
        }
    ],
    7: [  # Nitrogen
        {
            'shell_type': 's',
            'exponents': [99.1061690, 18.0523120, 4.8856602],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        },
        {
            'shell_type': 's',
            'exponents': [3.7804559, 0.8784966, 0.2857144],
            'coefficients': [-0.09996723, 0.39951283, 0.70011547]
        },
        {
            'shell_type': 'p',
            'exponents': [3.7804559, 0.8784966, 0.2857144],
            'coefficients': [0.15591627, 0.60768372, 0.39195739]
        }
    ],
    8: [  # Oxygen
        {
            'shell_type': 's',
            'exponents': [130.7093200, 23.8088610, 6.4436083],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        },
        {
            'shell_type': 's',
            'exponents': [5.0331513, 1.1695961, 0.3803890],
            'coefficients': [-0.09996723, 0.39951283, 0.70011547]
        },
        {
            'shell_type': 'p',
            'exponents': [5.0331513, 1.1695961, 0.3803890],
            'coefficients': [0.15591627, 0.60768372, 0.39195739]
        }
    ],
    9: [  # Fluorine
        {
            'shell_type': 's',
            'exponents': [166.6791300, 30.3608120, 8.2168207],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        },
        {
            'shell_type': 's',
            'exponents': [6.4648032, 1.5022812, 0.4885885],
            'coefficients': [-0.09996723, 0.39951283, 0.70011547]
        },
        {
            'shell_type': 'p',
            'exponents': [6.4648032, 1.5022812, 0.4885885],
            'coefficients': [0.15591627, 0.60768372, 0.39195739]
        }
    ],
    10: [  # Neon
        {
            'shell_type': 's',
            'exponents': [207.0156500, 37.7084590, 10.2053790],
            'coefficients': [0.15432897, 0.53532814, 0.44463454]
        },
        {
            'shell_type': 's',
            'exponents': [8.0246624, 1.8635050, 0.6059130],
            'coefficients': [-0.09996723, 0.39951283, 0.70011547]
        },
        {
            'shell_type': 'p',
            'exponents': [8.0246624, 1.8635050, 0.6059130],
            'coefficients': [0.15591627, 0.60768372, 0.39195739]
        }
    ],
}


BASIS_DATABASE = {
    'sto-3g': STO_3G_BASIS,
}


def get_basis_data(atomic_number: int, basis_name: str) -> List[Dict[str, Any]]:
    """
    Get basis set data for a specific atom.
    
    Args:
        atomic_number: Atomic number (Z)
        basis_name: Name of basis set (e.g., 'sto-3g')
    
    Returns:
        List of shell dictionaries with exponents and coefficients
    """
    basis_name = basis_name.lower()
    
    if basis_name not in BASIS_DATABASE:
        raise ValueError(f"Basis set '{basis_name}' not found. Available: {list(BASIS_DATABASE.keys())}")
    
    basis = BASIS_DATABASE[basis_name]
    
    if atomic_number not in basis:
        raise ValueError(f"Basis set '{basis_name}' not defined for Z={atomic_number}")
    
    return basis[atomic_number]


def load_basis(basis_name: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load entire basis set database.
    
    Args:
        basis_name: Name of basis set
    
    Returns:
        Dictionary mapping atomic numbers to basis data
    """
    basis_name = basis_name.lower()
    
    if basis_name not in BASIS_DATABASE:
        raise ValueError(f"Basis set '{basis_name}' not found")
    
    return BASIS_DATABASE[basis_name]


def add_basis_set(basis_name: str, basis_data: Dict[int, List[Dict[str, Any]]]):
    """
    Add a new basis set to the database.
    
    Args:
        basis_name: Name of basis set
        basis_data: Dictionary mapping atomic numbers to shell data
    """
    BASIS_DATABASE[basis_name.lower()] = basis_data


def list_available_basis_sets() -> List[str]:
    """List all available basis sets."""
    return list(BASIS_DATABASE.keys())
