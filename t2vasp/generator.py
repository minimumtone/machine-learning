"""
VASP input file generator — INCAR, POSCAR, KPOINTS, POTCAR script.

Given extracted entities and an intent, generates a complete set of
VASP input files in the specified output directory.
"""

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from t2vasp.entity import EntityResult, parse_formula
from t2vasp.intent import IntentResult
from t2vasp.templates import build_incar, format_incar

logger = logging.getLogger(__name__)


def _load_material_data(path: Optional[Path] = None) -> dict:
    if path is None:
        path = Path(__file__).parent / "config" / "material_terms.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Lattice constant estimation ──────────────────────────────────────

def estimate_lattice_constant(
    elements: List[str],
    counts: List[float],
    prototype: str,
    material_data: Optional[dict] = None,
) -> float:
    """Estimate lattice constant via Vegard's law.

    Parameters
    ----------
    elements : list of str
        Element symbols.
    counts : list of float
        Stoichiometric counts for each element.
    prototype : str
        Structure prototype (L12, B2, FCC, etc.).
    material_data : dict, optional
        Material terms data (loaded from YAML).

    Returns
    -------
    float
        Estimated lattice constant in Angstroms.
    """
    if material_data is None:
        material_data = _load_material_data()

    fcc_a0 = material_data.get("lattice_constants_fcc", {})
    total = sum(counts)
    fracs = [c / total for c in counts]

    a_avg = 0.0
    for elem, frac in zip(elements, fracs):
        a_elem = fcc_a0.get(elem, 3.80)
        a_avg += frac * a_elem

    # Prototype-dependent scaling (FCC reference → actual)
    proto_scale = {
        "BCC": 0.795,  # a_bcc ≈ a_fcc / 2^(1/3) ≈ 0.794 * a_fcc
        "B2": 0.795,
        "HCP": 0.707,  # a_hcp ≈ a_fcc / √2
    }
    scale = proto_scale.get(prototype, 1.0)
    return a_avg * scale


# ── POSCAR generators ────────────────────────────────────────────────

def _write_poscar_l12(path: Path, elements: List[str], counts: List[float],
                      a0: float) -> None:
    """Write L1₂ (Cu₃Au-type, Pm-3m) POSCAR."""
    # L12: face-center element (3×) + corner element (1×)
    if len(elements) < 2:
        raise ValueError("L1₂ requires at least 2 elements")
    el_face, el_corner = elements[0], elements[1]
    content = f"""{el_face}3{el_corner} L12 (Pm-3m)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el_face}  {el_corner}
  3  1
Direct
  0.000000  0.500000  0.500000
  0.500000  0.000000  0.500000
  0.500000  0.500000  0.000000
  0.000000  0.000000  0.000000
"""
    path.write_text(content)


def _write_poscar_b2(path: Path, elements: List[str], counts: List[float],
                     a0: float) -> None:
    """Write B2 (CsCl-type, Pm-3m) POSCAR."""
    if len(elements) < 2:
        raise ValueError("B2 requires at least 2 elements")
    el_a, el_b = elements[0], elements[1]
    content = f"""{el_a}{el_b} B2 (Pm-3m)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el_a}  {el_b}
  1  1
Direct
  0.000000  0.000000  0.000000
  0.500000  0.500000  0.500000
"""
    path.write_text(content)


def _write_poscar_fcc(path: Path, elements: List[str], counts: List[float],
                      a0: float) -> None:
    """Write FCC (A1) conventional cell."""
    el = elements[0]
    content = f"""{el} FCC (Fm-3m)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el}
  4
Direct
  0.000000  0.000000  0.000000
  0.500000  0.500000  0.000000
  0.500000  0.000000  0.500000
  0.000000  0.500000  0.500000
"""
    path.write_text(content)


def _write_poscar_bcc(path: Path, elements: List[str], counts: List[float],
                      a0: float) -> None:
    """Write BCC (A2) conventional cell."""
    el = elements[0]
    content = f"""{el} BCC (Im-3m)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el}
  2
Direct
  0.000000  0.000000  0.000000
  0.500000  0.500000  0.500000
"""
    path.write_text(content)


def _write_poscar_perovskite(path: Path, elements: List[str],
                             counts: List[float], a0: float) -> None:
    """Write perovskite ABO₃ POSCAR.

    Atom positions (cubic, Pm-3m):
      A: (0.5, 0.5, 0.5)
      B: (0.0, 0.0, 0.0)
      O: (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)
    """
    if len(elements) < 2:
        raise ValueError("Perovskite requires at least 2 cation elements (+O)")
    el_a, el_b = elements[0], elements[1]
    # Use O if present in composition, otherwise add it
    el_o = "O"
    content = f"""{el_a}{el_b}{el_o}3 perovskite (Pm-3m)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el_a}  {el_b}  {el_o}
  1  1  3
Direct
  0.500000  0.500000  0.500000
  0.000000  0.000000  0.000000
  0.500000  0.000000  0.000000
  0.000000  0.500000  0.000000
  0.000000  0.000000  0.500000
"""
    path.write_text(content)


def _write_poscar_rocksalt(path: Path, elements: List[str],
                           counts: List[float], a0: float) -> None:
    """Write rocksalt (NaCl, B1) POSCAR — 8 atoms (2×2×2 primitive)."""
    if len(elements) < 2:
        raise ValueError("Rocksalt requires 2 elements")
    el_m, el_x = elements[0], elements[1]
    content = f"""{el_m}{el_x} rocksalt (Fm-3m)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el_m}  {el_x}
  4  4
Direct
  0.000000  0.000000  0.000000
  0.500000  0.500000  0.000000
  0.500000  0.000000  0.500000
  0.000000  0.500000  0.500000
  0.500000  0.000000  0.000000
  0.000000  0.500000  0.000000
  0.000000  0.000000  0.500000
  0.500000  0.500000  0.500000
"""
    path.write_text(content)


def _write_poscar_rutile(path: Path, elements: List[str],
                         counts: List[float], a0: float) -> None:
    """Write rutile TiO₂-type POSCAR (P4₂/mnm)."""
    if len(elements) < 2:
        raise ValueError("Rutile requires 2 elements")
    el_m, el_x = elements[0], elements[1]
    c = a0 * 0.644  # c/a ≈ 0.644 for rutile TiO₂
    u = 0.305  # internal parameter
    content = f"""{el_m}{el_x}2 rutile (P42/mnm)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {c:.6f}
  {el_m}  {el_x}
  2  4
Direct
  0.000000  0.000000  0.000000
  0.500000  0.500000  0.500000
  {u:.6f}  {u:.6f}  0.000000
  {1-u:.6f}  {1-u:.6f}  0.000000
  {0.5+u:.6f}  {0.5-u:.6f}  0.500000
  {0.5-u:.6f}  {0.5+u:.6f}  0.500000
"""
    path.write_text(content)


def _write_poscar_wurtzite(path: Path, elements: List[str],
                           counts: List[float], a0: float) -> None:
    """Write wurtzite (B4) POSCAR (P6₃mc)."""
    if len(elements) < 2:
        raise ValueError("Wurtzite requires 2 elements")
    el_m, el_x = elements[0], elements[1]
    c = a0 * 1.633  # ideal c/a
    u = 0.375
    content = f"""{el_m}{el_x} wurtzite (P63mc)
1.0
  {a0:.6f}  0.000000  0.000000
  {-a0/2:.6f}  {a0*math.sqrt(3)/2:.6f}  0.000000
  0.000000  0.000000  {c:.6f}
  {el_m}  {el_x}
  2  2
Direct
  0.333333  0.666667  0.000000
  0.666667  0.333333  0.500000
  0.333333  0.666667  {u:.6f}
  0.666667  0.333333  {0.5+u:.6f}
"""
    path.write_text(content)


def _write_poscar_zincblende(path: Path, elements: List[str],
                             counts: List[float], a0: float) -> None:
    """Write zincblende (B3) POSCAR (F-43m)."""
    if len(elements) < 2:
        raise ValueError("Zincblende requires 2 elements")
    el_m, el_x = elements[0], elements[1]
    content = f"""{el_m}{el_x} zincblende (F-43m)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el_m}  {el_x}
  4  4
Direct
  0.000000  0.000000  0.000000
  0.500000  0.500000  0.000000
  0.500000  0.000000  0.500000
  0.000000  0.500000  0.500000
  0.250000  0.250000  0.250000
  0.750000  0.750000  0.250000
  0.750000  0.250000  0.750000
  0.250000  0.750000  0.750000
"""
    path.write_text(content)


def _write_poscar_hcp(path: Path, elements: List[str],
                      counts: List[float], a0: float) -> None:
    """Write HCP (A3) POSCAR."""
    el = elements[0]
    c = a0 * 1.633
    content = f"""{el} HCP (P63/mmc)
1.0
  {a0:.6f}  0.000000  0.000000
  {-a0/2:.6f}  {a0*math.sqrt(3)/2:.6f}  0.000000
  0.000000  0.000000  {c:.6f}
  {el}
  2
Direct
  0.333333  0.666667  0.250000
  0.666667  0.333333  0.750000
"""
    path.write_text(content)


def _write_poscar_diamond(path: Path, elements: List[str],
                          counts: List[float], a0: float) -> None:
    """Write diamond (A4) POSCAR (Fd-3m)."""
    el = elements[0]
    content = f"""{el} diamond (Fd-3m)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el}
  8
Direct
  0.000000  0.000000  0.000000
  0.500000  0.500000  0.000000
  0.500000  0.000000  0.500000
  0.000000  0.500000  0.500000
  0.250000  0.250000  0.250000
  0.750000  0.750000  0.250000
  0.750000  0.250000  0.750000
  0.250000  0.750000  0.750000
"""
    path.write_text(content)


# POSCAR writer dispatch
_POSCAR_WRITERS = {
    "L12": _write_poscar_l12,
    "B2": _write_poscar_b2,
    "FCC": _write_poscar_fcc,
    "BCC": _write_poscar_bcc,
    "HCP": _write_poscar_hcp,
    "perovskite": _write_poscar_perovskite,
    "rocksalt": _write_poscar_rocksalt,
    "rutile": _write_poscar_rutile,
    "wurtzite": _write_poscar_wurtzite,
    "zincblende": _write_poscar_zincblende,
    "diamond": _write_poscar_diamond,
}


def _infer_prototype(entity: EntityResult) -> str:
    """Infer structure prototype from composition when not explicitly given."""
    comp = entity.composition
    if not comp:
        return "FCC"
    elements = list(comp.keys())
    counts = list(comp.values())
    n_elem = len(elements)

    # Check for oxygen-containing → oxide prototypes
    if "O" in comp:
        o_count = comp["O"]
        cations = {k: v for k, v in comp.items() if k != "O"}
        n_cat = len(cations)
        if n_cat == 2 and abs(o_count - 3.0) < 0.1:
            return "perovskite"
        if n_cat == 1:
            cat_count = list(cations.values())[0]
            ratio = o_count / cat_count
            if abs(ratio - 2.0) < 0.1:
                return "rutile"
            if abs(ratio - 1.0) < 0.1:
                return "rocksalt"

    if n_elem == 1:
        return "FCC"  # default for single element
    if n_elem == 2:
        ratio = max(counts) / min(counts)
        if abs(ratio - 3.0) < 0.1:
            return "L12"
        if abs(ratio - 1.0) < 0.1:
            return "B2"

    return "FCC"


# ── KPOINTS ──────────────────────────────────────────────────────────

def write_kpoints(path: Path, mesh: Tuple[int, int, int] = (12, 12, 12)) -> None:
    """Write KPOINTS file (Gamma-centered)."""
    content = f"""Automatic mesh
0
Gamma
  {mesh[0]} {mesh[1]} {mesh[2]}
  0 0 0
"""
    path.write_text(content)


def estimate_kpoints(a0: float, prototype: str) -> Tuple[int, int, int]:
    """Estimate k-point mesh from lattice constant.

    Rule: k_i × a_i ≈ 30–40 Å for metals.
    """
    target_ka = 35.0
    k = max(1, round(target_ka / a0))
    # For hexagonal structures, adjust kz
    if prototype in ("HCP", "wurtzite", "rutile"):
        c = a0 * 1.633 if prototype in ("HCP", "wurtzite") else a0 * 0.644
        kz = max(1, round(target_ka / c))
        return (k, k, kz)
    return (k, k, k)


# ── POTCAR script ────────────────────────────────────────────────────

def write_potcar_script(path: Path, elements: List[str],
                        material_data: Optional[dict] = None) -> None:
    """Write a shell script that generates POTCAR from $VASPPOT."""
    if material_data is None:
        material_data = _load_material_data()
    potcar_variants = material_data.get("potcar_variants", {})

    lines = [
        "#!/bin/bash",
        "# POTCAR generation script (auto-generated by t2vasp)",
        "# Usage: bash make_potcar.sh",
        "#",
        "# Requires: $VASPPOT pointing to PAW-PBE pseudopotential directory",
        "#   e.g., export VASPPOT=/path/to/potpaw_PBE.64",
        "",
        'if [ -z "$VASPPOT" ]; then',
        '    echo "Error: VASPPOT environment variable is not set."',
        '    echo "Set it to the PAW-PBE pseudopotential directory, e.g.:"',
        '    echo "  export VASPPOT=/path/to/potpaw_PBE.64"',
        '    exit 1',
        'fi',
        "",
    ]

    cat_parts = []
    for elem in elements:
        variant = potcar_variants.get(elem, elem)
        cat_parts.append(f'"$VASPPOT"/{variant}/POTCAR')
        lines.append(f'echo "  {elem} -> {variant}"')

    lines.append("")
    lines.append("cat " + " ".join(cat_parts) + " > POTCAR")
    lines.append('echo "POTCAR generated successfully."')
    lines.append("")

    path.write_text("\n".join(lines))
    path.chmod(0o755)


# ── Main generation pipeline ────────────────────────────────────────

def generate(
    intent: IntentResult,
    entity: EntityResult,
    output_dir: Path,
    scheduler: str = "slurm",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Generate VASP input files from intent and entity results.

    Parameters
    ----------
    intent : IntentResult
        Classified calculation intent.
    entity : EntityResult
        Extracted material entities.
    output_dir : Path
        Directory to write files into.
    scheduler : str
        Job scheduler type: "slurm", "pbs", or "local".
    dry_run : bool
        If True, return plan without writing files.

    Returns
    -------
    dict
        Generation plan and file paths.
    """
    material_data = _load_material_data()

    # Determine elements and composition
    elements = entity.species_list
    if not elements:
        raise ValueError("No elements found in query. Please specify a formula "
                         "or element names.")

    counts = [entity.composition.get(el, 1.0) for el in elements]

    # Determine prototype
    prototype = entity.prototype or _infer_prototype(entity)

    # Estimate lattice constant
    a0 = estimate_lattice_constant(elements, counts, prototype, material_data)

    # Estimate k-points
    kpoints = entity.kpoints or estimate_kpoints(a0, prototype)

    # Build INCAR
    formula = entity.formula_str or "".join(
        f"{el}{int(c)}" if c != 1 else el for el, c in zip(elements, counts)
    )
    incar = build_incar(
        calc_type=intent.calc_type,
        formula=formula,
        spin_polarized=entity.spin_polarized,
        encut=entity.encut,
    )

    # Build generation plan
    plan = {
        "query": entity.raw_query,
        "calc_type": intent.calc_type,
        "formula": formula,
        "elements": elements,
        "composition": dict(zip(elements, counts)),
        "prototype": prototype,
        "lattice_constant_angstrom": round(a0, 4),
        "kpoints": list(kpoints),
        "spin_polarized": bool(incar.get("ISPIN", 1) == 2),
        "output_dir": str(output_dir),
        "scheduler": scheduler,
        "files": ["INCAR", "POSCAR", "KPOINTS", "make_potcar.sh"],
        "secondary_steps": intent.secondary_types,
    }

    if scheduler in ("slurm", "pbs"):
        plan["files"].append(f"job_{scheduler}.sh")

    plan["files"].append("t2vasp_plan.yaml")

    logger.info("Generation plan: %s", plan)

    if dry_run:
        return plan

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write INCAR
    incar_path = output_dir / "INCAR"
    incar_path.write_text(format_incar(incar))
    logger.info("Wrote %s", incar_path)

    # Write POSCAR
    poscar_path = output_dir / "POSCAR"
    writer = _POSCAR_WRITERS.get(prototype)
    if writer is None:
        logger.warning("No POSCAR writer for prototype '%s'; using FCC", prototype)
        writer = _write_poscar_fcc
    writer(poscar_path, elements, counts, a0)
    logger.info("Wrote %s", poscar_path)

    # Write KPOINTS
    kpoints_path = output_dir / "KPOINTS"
    write_kpoints(kpoints_path, kpoints)
    logger.info("Wrote %s", kpoints_path)

    # Write POTCAR script
    potcar_script_path = output_dir / "make_potcar.sh"
    write_potcar_script(potcar_script_path, elements, material_data)
    logger.info("Wrote %s", potcar_script_path)

    # Write job script
    from t2vasp.scheduler import generate_job_script
    generate_job_script(
        output_dir,
        scheduler=scheduler,
        job_name=f"t2vasp_{formula}",
    )

    # Write plan
    plan_path = output_dir / "t2vasp_plan.yaml"
    with open(plan_path, "w", encoding="utf-8") as f:
        yaml.dump(plan, f, default_flow_style=False, allow_unicode=True)
    logger.info("Wrote %s", plan_path)

    return plan
