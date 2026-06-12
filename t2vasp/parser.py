"""
VASP output-file parser.

Extracts structured data from OUTCAR, vasprun.xml, POSCAR/CONTCAR,
and DOSCAR without external VASP-specific libraries — only stdlib
``xml.etree`` and numpy are required.
"""

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Data containers ──────────────────────────────────────────────────
@dataclass
class StructureData:
    """Lattice vectors, species, and fractional coordinates."""
    lattice: np.ndarray              # (3, 3) in Angstrom
    species: List[str]
    positions: np.ndarray            # (N, 3) fractional
    scale_factor: float = 1.0

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.lattice)))

    @property
    def lattice_constant(self) -> float:
        """Magnitude of the first lattice vector (cubic approximation)."""
        return float(np.linalg.norm(self.lattice[0]))


@dataclass
class OutcarData:
    """Key quantities parsed from an OUTCAR file."""
    total_energy: Optional[float] = None       # eV (TOTEN)
    energy_without_entropy: Optional[float] = None  # eV (sigma->0)
    num_atoms: int = 0
    converged: bool = False
    ionic_steps: int = 0
    forces: Optional[np.ndarray] = None        # (N, 3)
    stress_tensor: Optional[np.ndarray] = None # (6,) Voigt
    magnetization: Optional[float] = None
    elapsed_time: Optional[float] = None       # seconds
    fermi_energy: Optional[float] = None
    # Orbital-projected quantities (populated if LORBIT >= 10)
    orbital_moments: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def energy_per_atom(self) -> Optional[float]:
        if self.total_energy is not None and self.num_atoms > 0:
            return self.total_energy / self.num_atoms
        return None


@dataclass
class DosData:
    """Density-of-states arrays parsed from DOSCAR or vasprun.xml."""
    energies: np.ndarray               # (nE,)
    total_dos: np.ndarray              # (nE,) or (nE, 2) for spin-polarised
    fermi_energy: float = 0.0
    projected_dos: Optional[Dict[str, np.ndarray]] = None  # atom/orbital → (nE,)


@dataclass
class VasprunData:
    """Aggregated data from vasprun.xml."""
    program_version: str = ""
    parameters: Dict[str, str] = field(default_factory=dict)
    initial_structure: Optional[StructureData] = None
    final_structure: Optional[StructureData] = None
    total_energy: Optional[float] = None
    fermi_energy: Optional[float] = None
    converged_electronic: bool = False
    converged_ionic: bool = False
    dos: Optional[DosData] = None
    eigenvalues: Optional[np.ndarray] = None  # (nkpt, nbands)


# ── POSCAR / CONTCAR parser ─────────────────────────────────────────
def parse_poscar(path: str | Path) -> StructureData:
    """Parse POSCAR / CONTCAR into :class:`StructureData`."""
    path = Path(path)
    lines = path.read_text().splitlines()
    if len(lines) < 8:
        raise ValueError(f"File too short to be a valid POSCAR: {path}")

    scale = float(lines[1].strip())
    lattice = np.array(
        [[float(x) for x in lines[i].split()] for i in range(2, 5)]
    )
    if scale < 0:
        # Negative scale → volume specification (rare)
        vol = abs(scale)
        scale = (vol / abs(np.linalg.det(lattice))) ** (1 / 3)
    lattice *= scale

    # Species line (VASP 5+) and counts
    species_line = lines[5].split()
    try:
        counts = [int(x) for x in species_line]
        species_names: List[str] = []
    except ValueError:
        species_names = species_line
        counts = [int(x) for x in lines[6].split()]

    species: List[str] = []
    for name, cnt in zip(
        species_names if species_names else [f"X{i}" for i in range(len(counts))],
        counts,
    ):
        species.extend([name] * cnt)

    # Coordinate block starts after 'Selective dynamics' (optional) + coord type
    coord_start = 7 if species_names else 6
    if lines[coord_start].strip()[0] in ("S", "s"):
        coord_start += 1  # skip Selective dynamics
    coord_start += 1  # skip Direct/Cartesian line

    natoms = sum(counts)
    positions = np.zeros((natoms, 3))
    for i in range(natoms):
        positions[i] = [float(x) for x in lines[coord_start + i].split()[:3]]

    return StructureData(
        lattice=lattice,
        species=species,
        positions=positions,
        scale_factor=1.0,  # already applied
    )


# ── OUTCAR parser ───────────────────────────────────────────────────
def parse_outcar(path: str | Path) -> OutcarData:
    """Extract key quantities from OUTCAR."""
    path = Path(path)
    data = OutcarData()
    text = path.read_text(errors="replace")

    # Number of atoms
    m = re.search(r"NIONS\s*=\s*(\d+)", text)
    if m:
        data.num_atoms = int(m.group(1))

    # Energies (last occurrence)
    for m in re.finditer(r"free  energy   TOTEN\s*=\s*([-\d.Ee+]+)", text):
        data.total_energy = float(m.group(1))
    for m in re.finditer(r"energy  without entropy\s*=\s*([-\d.Ee+]+)", text):
        data.energy_without_entropy = float(m.group(1))

    # Fermi energy
    m_f = re.search(r"E-fermi\s*:\s*([-\d.Ee+]+)", text)
    if m_f:
        data.fermi_energy = float(m_f.group(1))

    # Convergence
    data.converged = "reached required accuracy" in text

    # Ionic steps
    data.ionic_steps = text.count("FREE ENERGIE OF THE ION-ELECTRON SYSTEM")

    # Magnetization
    for m in re.finditer(r"number of electron\s+[-\d.]+\s+magnetization\s+([-\d.Ee+]+)", text):
        data.magnetization = float(m.group(1))

    # Elapsed time
    m_t = re.search(r"Elapsed time \(sec\):\s*([\d.]+)", text)
    if m_t:
        data.elapsed_time = float(m_t.group(1))

    # Forces (last block)
    force_blocks = list(re.finditer(
        r"TOTAL-FORCE \(eV/Angst\)\s*\n\s*-+\n((?:\s*[-\d.Ee+]+\s+[-\d.Ee+]+\s+"
        r"[-\d.Ee+]+\s+[-\d.Ee+]+\s+[-\d.Ee+]+\s+[-\d.Ee+]+\s*\n)+)",
        text,
    ))
    if force_blocks:
        block = force_blocks[-1].group(1)
        rows = [line.split() for line in block.strip().splitlines()]
        data.forces = np.array([[float(r[3]), float(r[4]), float(r[5])] for r in rows])

    # Stress tensor (last occurrence, kB units)
    stress_matches = list(re.finditer(
        r"in kB\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)"
        r"\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)",
        text,
    ))
    if stress_matches:
        vals = stress_matches[-1].groups()
        data.stress_tensor = np.array([float(v) for v in vals])

    logger.info("Parsed OUTCAR: E=%.4f eV, %d atoms, converged=%s",
                data.total_energy or float("nan"), data.num_atoms, data.converged)
    return data


# ── DOSCAR parser ───────────────────────────────────────────────────
def parse_doscar(path: str | Path) -> DosData:
    """Parse a DOSCAR file (total DOS only; projected DOS via vasprun.xml)."""
    path = Path(path)
    lines = path.read_text().splitlines()

    header = lines[5].split()
    e_max, e_min, ndos = float(header[0]), float(header[1]), int(header[2])
    e_fermi = float(header[3])

    energies = np.zeros(ndos)
    dos_vals = np.zeros(ndos)
    for i in range(ndos):
        parts = lines[6 + i].split()
        energies[i] = float(parts[0])
        dos_vals[i] = float(parts[1])

    return DosData(energies=energies, total_dos=dos_vals, fermi_energy=e_fermi)


# ── vasprun.xml parser ──────────────────────────────────────────────
def parse_vasprun(path: str | Path) -> VasprunData:
    """Parse vasprun.xml using ElementTree (stdlib, no lxml needed)."""
    path = Path(path)
    data = VasprunData()

    try:
        tree = ET.parse(str(path))
    except ET.ParseError as exc:
        logger.warning("Failed to parse vasprun.xml: %s", exc)
        return data

    root = tree.getroot()

    # Program version
    gen = root.find(".//generator/i[@name='program']")
    if gen is not None and gen.text:
        data.program_version = gen.text.strip()

    # Parameters (selected subset)
    for tag in root.findall(".//parameters//i"):
        name = tag.get("name", "")
        if name and tag.text:
            data.parameters[name] = tag.text.strip()

    # Final energy
    energies = root.findall(".//calculation/energy/i[@name='e_fr_energy']")
    if energies:
        data.total_energy = float(energies[-1].text.strip())

    # Fermi energy
    dos_el = root.find(".//calculation/dos/i[@name='efermi']")
    if dos_el is not None and dos_el.text:
        data.fermi_energy = float(dos_el.text.strip())

    # Convergence
    conv_el = root.findall(".//calculation/scstep")
    conv_ionic = root.findall(".//calculation")
    if conv_el:
        data.converged_electronic = True
    if conv_ionic:
        data.converged_ionic = True

    # Structures (initial and final)
    structs = root.findall(".//structure")
    if structs:
        data.initial_structure = _parse_xml_structure(structs[0])
        data.final_structure = _parse_xml_structure(structs[-1])

    # DOS
    dos_block = root.find(".//calculation/dos/total/array")
    if dos_block is not None:
        data.dos = _parse_xml_dos(dos_block, data.fermi_energy or 0.0)

    logger.info("Parsed vasprun.xml: E=%.4f eV, version=%s",
                data.total_energy or float("nan"), data.program_version)
    return data


def _parse_xml_structure(struct_el: ET.Element) -> StructureData:
    """Extract lattice + positions from a <structure> element."""
    basis = struct_el.find(".//crystal/varray[@name='basis']")
    lattice = np.zeros((3, 3))
    if basis is not None:
        for i, v in enumerate(basis.findall("v")):
            lattice[i] = [float(x) for x in v.text.split()]

    pos_el = struct_el.find(".//varray[@name='positions']")
    species: List[str] = []
    positions_list: List[List[float]] = []
    if pos_el is not None:
        for v in pos_el.findall("v"):
            positions_list.append([float(x) for x in v.text.split()])

    positions = np.array(positions_list) if positions_list else np.zeros((0, 3))
    return StructureData(lattice=lattice, species=species, positions=positions)


def _parse_xml_dos(array_el: ET.Element, fermi: float) -> DosData:
    """Extract total DOS from a <total><array> element."""
    set_el = array_el.find(".//set/set")
    if set_el is None:
        return DosData(energies=np.array([]), total_dos=np.array([]),
                       fermi_energy=fermi)

    rows = []
    for r in set_el.findall("r"):
        rows.append([float(x) for x in r.text.split()])
    arr = np.array(rows)
    return DosData(
        energies=arr[:, 0] if arr.size and arr.ndim > 1 else np.array([]),
        total_dos=arr[:, 1] if arr.ndim > 1 and arr.shape[1] > 1 else np.array([]),
        fermi_energy=fermi,
    )


# ── Convenience: parse an entire calc directory ─────────────────────
def parse_calc_dir(calc_dir: str | Path) -> Dict:
    """Parse all available VASP files in *calc_dir* and return a dict.

    Keys present depend on which files exist:
    ``outcar``, ``structure``, ``vasprun``, ``dos``.
    """
    d = Path(calc_dir)
    result: Dict = {"path": str(d)}

    contcar = d / "CONTCAR"
    poscar = d / "POSCAR"
    if contcar.is_file() and contcar.stat().st_size > 0:
        result["structure"] = parse_poscar(contcar)
    elif poscar.is_file():
        result["structure"] = parse_poscar(poscar)

    outcar = d / "OUTCAR"
    if outcar.is_file():
        result["outcar"] = parse_outcar(outcar)

    vasprun = d / "vasprun.xml"
    if vasprun.is_file():
        result["vasprun"] = parse_vasprun(vasprun)

    doscar = d / "DOSCAR"
    if doscar.is_file():
        result["dos"] = parse_doscar(doscar)

    return result
