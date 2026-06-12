"""
INCAR templates — calculation-type-specific VASP parameter sets.

Each template is a dict of INCAR tags.  The ``build_incar`` function
merges a base template with calc-type-specific overrides and any
user-supplied parameters.
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Base template (shared by all calculation types) ───────────────────

_BASE: Dict[str, Any] = {
    "SYSTEM": "{formula} calculation",
    # Electronic
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-6,
    "NELM": 200,
    "LREAL": ".FALSE.",
    # Smearing (metals by default)
    "ISMEAR": 1,
    "SIGMA": 0.2,
    # Exchange-correlation
    "GGA": "PE",
    # Output
    "LORBIT": 11,
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
    # Performance
    "NCORE": 4,
}

# ── Per-calc-type overrides ──────────────────────────────────────────

INCAR_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "relax": {
        "IBRION": 2,
        "ISIF": 3,
        "NSW": 100,
        "EDIFFG": -0.01,
        "LCHARG": ".TRUE.",   # keep CHGCAR for follow-up static/DOS
    },
    "static": {
        "IBRION": -1,
        "NSW": 0,
        "LCHARG": ".TRUE.",
    },
    "dos": {
        "IBRION": -1,
        "NSW": 0,
        "ISMEAR": -5,       # tetrahedron method for accurate DOS
        "NEDOS": 2001,
        "ICHARG": 11,       # read charge from CHGCAR
        "LCHARG": ".TRUE.",
    },
    "band": {
        "IBRION": -1,
        "NSW": 0,
        "ICHARG": 11,
        "LORBIT": 11,
        "LCHARG": ".FALSE.",
    },
    "crystal_field": {
        "IBRION": 2,
        "ISIF": 3,
        "NSW": 100,
        "EDIFFG": -0.01,
        "ISPIN": 2,
        "LORBIT": 11,
        "LCHARG": ".TRUE.",
    },
    "phonon": {
        "IBRION": 6,         # finite differences
        "ISIF": 0,
        "NSW": 1,
        "NFREE": 2,
        "EDIFF": 1e-8,       # tight convergence for forces
        "PREC": "Accurate",
        "LREAL": ".FALSE.",
        "ADDGRID": ".TRUE.",
    },
    "elastic": {
        "IBRION": 6,
        "ISIF": 3,
        "NSW": 1,
        "NFREE": 4,
        "EDIFF": 1e-7,
        "PREC": "Accurate",
    },
    "polarization": {
        "IBRION": -1,
        "NSW": 0,
        "LCALCPOL": ".TRUE.",
        "DIPOL": "0.5 0.5 0.5",
        "IDIPOL": 0,
        "LCHARG": ".TRUE.",
    },
    "dielectric": {
        "IBRION": -1,
        "NSW": 0,
        "LEPSILON": ".TRUE.",
        "LRPA": ".FALSE.",
        "EDIFF": 1e-8,
    },
    "magnetic": {
        "IBRION": 2,
        "ISIF": 3,
        "NSW": 100,
        "EDIFFG": -0.01,
        "ISPIN": 2,
        "LCHARG": ".TRUE.",
    },
    "neb": {
        "IBRION": 3,
        "POTIM": 0.0,
        "ISIF": 0,
        "NSW": 200,
        "EDIFFG": -0.03,
        "SPRING": -5,
        "LCLIMB": ".TRUE.",
        "IMAGES": 5,
        "LCHARG": ".FALSE.",
        "LWAVE": ".FALSE.",
    },
    "md": {
        "IBRION": 0,
        "NSW": 5000,
        "POTIM": 1.0,        # 1 fs time step
        "SMASS": 0,           # NVT Nose-Hoover
        "TEBEG": 300,
        "TEEND": 300,
        "ISIF": 2,
        "PREC": "Normal",
        "LWAVE": ".FALSE.",
        "LCHARG": ".FALSE.",
    },
    "sqs": {
        # SQS uses relax settings
        "IBRION": 2,
        "ISIF": 3,
        "NSW": 100,
        "EDIFFG": -0.01,
        "ISPIN": 2,
        "LCHARG": ".TRUE.",
    },
}


def build_incar(
    calc_type: str,
    formula: str = "material",
    spin_polarized: Optional[bool] = None,
    encut: Optional[int] = None,
    is_metal: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a complete INCAR parameter dictionary.

    Parameters
    ----------
    calc_type : str
        One of INCAR_TEMPLATES keys.
    formula : str
        Chemical formula for SYSTEM tag.
    spin_polarized : bool, optional
        Force ISPIN=2 if True.
    encut : int, optional
        Override ENCUT.
    is_metal : bool
        If False, use ISMEAR=0 instead of 1 (for insulators/semiconductors).
    overrides : dict, optional
        Arbitrary INCAR tag overrides.

    Returns
    -------
    dict
        Complete INCAR key-value pairs.
    """
    incar = deepcopy(_BASE)
    incar["SYSTEM"] = f"{formula} {calc_type} calculation"

    # Merge calc-type template
    ct_template = INCAR_TEMPLATES.get(calc_type, {})
    incar.update(ct_template)

    # Spin polarization
    if spin_polarized:
        incar["ISPIN"] = 2

    # ENCUT override
    if encut is not None:
        incar["ENCUT"] = encut

    # Semiconductor/insulator adjustment
    if not is_metal and incar.get("ISMEAR") == 1:
        incar["ISMEAR"] = 0
        incar["SIGMA"] = 0.05

    # User overrides
    if overrides:
        incar.update(overrides)

    logger.info("Built INCAR for %s/%s: %d tags", calc_type, formula, len(incar))
    return incar


def format_incar(incar: Dict[str, Any]) -> str:
    """Format INCAR dict as VASP-readable text.

    Groups related tags with comment headers for readability.
    """
    # Define tag grouping order
    groups = [
        ("System", ["SYSTEM"]),
        ("Electronic relaxation", [
            "ENCUT", "PREC", "EDIFF", "NELM", "ALGO", "LREAL", "ADDGRID",
        ]),
        ("Ionic relaxation", [
            "IBRION", "ISIF", "NSW", "EDIFFG", "POTIM", "NFREE",
        ]),
        ("Smearing", ["ISMEAR", "SIGMA"]),
        ("Exchange-correlation", ["GGA"]),
        ("Spin", ["ISPIN", "MAGMOM"]),
        ("Molecular dynamics", [
            "SMASS", "TEBEG", "TEEND",
        ]),
        ("Polarization / Dielectric", [
            "LCALCPOL", "DIPOL", "IDIPOL", "LEPSILON", "LRPA",
        ]),
        ("NEB", ["IMAGES", "SPRING", "LCLIMB"]),
        ("Output", ["LORBIT", "NEDOS", "ICHARG", "LWAVE", "LCHARG"]),
        ("Performance", ["NCORE", "KPAR", "NPAR"]),
    ]

    lines: list[str] = []
    used_keys: set[str] = set()

    for group_name, tags in groups:
        group_lines: list[str] = []
        for tag in tags:
            if tag in incar:
                val = incar[tag]
                group_lines.append(f"{tag} = {val}")
                used_keys.add(tag)
        if group_lines:
            lines.append(f"# {group_name}")
            lines.extend(group_lines)
            lines.append("")

    # Remaining tags not in any group
    remaining = {k: v for k, v in incar.items() if k not in used_keys}
    if remaining:
        lines.append("# Additional")
        for k, v in remaining.items():
            lines.append(f"{k} = {v}")
        lines.append("")

    return "\n".join(lines)
