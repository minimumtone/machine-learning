"""Generate seed CSV files for the L1_2 materials database.

Running this script writes seven CSV files into ``db/seed/`` that can be
loaded into PostgreSQL after the schema has been applied.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CompoundSpec:
    formula: str
    a_element: str
    b_element: str
    lattice_a: float
    formation_energy: float
    energy_above_hull: float
    is_stable: bool
    bulk_modulus: float
    shear_modulus: float


KNOWN_L12_COMPOUNDS: list[CompoundSpec] = [
    CompoundSpec("Ni3Al", "Ni", "Al", 3.572, -0.420, 0.000, True, 180.0, 85.0),
    CompoundSpec("Ni3Ga", "Ni", "Ga", 3.660, -0.380, 0.010, True, 165.0, 74.0),
    CompoundSpec("Ni3Ge", "Ni", "Ge", 3.610, -0.350, 0.040, False, 175.0, 78.0),
    CompoundSpec("Co3Ti", "Co", "Ti", 3.550, -0.450, 0.000, True, 190.0, 90.0),
    CompoundSpec("Al3Sc", "Al", "Sc", 4.090, -0.500, 0.000, True, 155.0, 65.0),
    CompoundSpec("Al3Ti", "Al", "Ti", 3.980, -0.410, 0.020, True, 160.0, 70.0),
    CompoundSpec("Pt3Al", "Pt", "Al", 3.900, -0.550, 0.000, True, 210.0, 95.0),
    CompoundSpec("Ir3Nb", "Ir", "Nb", 3.870, -0.480, 0.030, False, 220.0, 100.0),
    CompoundSpec("Co3Al", "Co", "Al", 3.670, -0.360, 0.010, True, 180.0, 82.0),
    CompoundSpec("Co3W", "Co", "W", 3.740, -0.330, 0.050, False, 200.0, 88.0),
    CompoundSpec("Co3Ta", "Co", "Ta", 3.720, -0.340, 0.040, False, 198.0, 86.0),
]

ADDITIONAL_A = [
    "Ni", "Co", "Fe", "Cu", "Pd", "Pt", "Ir", "Rh", "Ru", "Ag", "Au",
]
ADDITIONAL_B = [
    "Al", "Ga", "Ge", "Ti", "Nb", "Ta", "Sc", "Y", "Hf", "Si", "Sn",
    "Zn", "V", "Zr", "Mn",
]


def _chemical_system(a: str, b: str) -> str:
    return "-".join(sorted({a, b}))


def generate_compounds(target_count: int = 120) -> list[CompoundSpec]:
    compounds: list[CompoundSpec] = list(KNOWN_L12_COMPOUNDS)
    seen = {c.formula for c in compounds}
    idx = 0
    for a in ADDITIONAL_A:
        for b in ADDITIONAL_B:
            formula = f"{a}3{b}"
            if formula in seen:
                continue
            if len(compounds) >= target_count:
                return compounds[:target_count]
            lattice_a = round(3.40 + 0.02 * (idx % 35), 3)
            fe = round(-0.30 - 0.005 * (idx % 40), 3)
            eah = round(0.004 * (idx % 14), 3)
            is_stable = eah <= 0.020
            bm = 140 + (idx % 15) * 6
            sm = 55 + (idx % 12) * 5
            compounds.append(
                CompoundSpec(formula, a, b, lattice_a, fe, eah, is_stable, bm, sm)
            )
            seen.add(formula)
            idx += 1
    return compounds[:target_count]


@dataclass(frozen=True)
class SeedRecord:
    entry_id: str
    source_id: str
    formula: str
    a_element: str
    b_element: str
    lattice_a: float
    formation_energy: float
    energy_above_hull: float
    is_stable: bool
    bulk_modulus: float
    shear_modulus: float

    @property
    def chemical_system(self) -> str:
        return _chemical_system(self.a_element, self.b_element)


def build_seed_records(target_count: int = 120) -> list[SeedRecord]:
    records: list[SeedRecord] = []
    for i, c in enumerate(generate_compounds(target_count), start=1):
        eid = f"entry_{c.formula}_{i:03d}"
        records.append(
            SeedRecord(
                entry_id=eid,
                source_id=f"mock_{c.formula}_{i:03d}",
                formula=c.formula,
                a_element=c.a_element,
                b_element=c.b_element,
                lattice_a=c.lattice_a,
                formation_energy=c.formation_energy,
                energy_above_hull=c.energy_above_hull,
                is_stable=c.is_stable,
                bulk_modulus=c.bulk_modulus,
                shear_modulus=c.shear_modulus,
            )
        )
    return records


def _write_csv(
    path: Path,
    header: Iterable[str],
    rows: Iterable[Iterable[object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(header))
        w.writerows(rows)


def generate_seed_files(base_dir: Path, target_count: int = 120) -> None:
    records = build_seed_records(target_count)

    _write_csv(
        base_dir / "seed_l12_entries.csv",
        [
            "entry_id", "source_db", "source_material_id", "formula",
            "reduced_formula", "chemical_system", "number_of_elements",
        ],
        [
            [r.entry_id, "mock", r.source_id, r.formula, r.formula,
             r.chemical_system, 2]
            for r in records
        ],
    )

    comp_rows: list[list[object]] = []
    for r in records:
        comp_rows.append([f"comp_{r.entry_id}_A", r.entry_id, r.a_element, 0.75, "A"])
        comp_rows.append([f"comp_{r.entry_id}_B", r.entry_id, r.b_element, 0.25, "B"])
    _write_csv(
        base_dir / "seed_composition.csv",
        ["composition_id", "entry_id", "element", "atomic_fraction", "site_label"],
        comp_rows,
    )

    _write_csv(
        base_dir / "seed_structure.csv",
        [
            "structure_id", "entry_id", "prototype", "strukturbericht",
            "formula_type", "space_group_number", "crystal_system",
            "lattice_a", "lattice_b", "lattice_c", "volume_per_atom",
        ],
        [
            [
                f"struct_{r.entry_id}", r.entry_id, "L12", "L12", "A3B",
                221, "cubic", r.lattice_a, r.lattice_a, r.lattice_a,
                round(r.lattice_a ** 3 / 4, 2),
            ]
            for r in records
        ],
    )

    _write_csv(
        base_dir / "seed_phase_stability.csv",
        [
            "stability_id", "entry_id", "formation_energy_per_atom",
            "energy_above_hull", "is_stable",
        ],
        [
            [
                f"stab_{r.entry_id}", r.entry_id, r.formation_energy,
                r.energy_above_hull, str(r.is_stable).lower(),
            ]
            for r in records
        ],
    )

    _write_csv(
        base_dir / "seed_calculation.csv",
        ["calculation_id", "entry_id", "method", "functional", "calculation_type"],
        [[f"calc_{r.entry_id}", r.entry_id, "DFT", "PBE", "static"]
         for r in records],
    )

    prop_rows: list[list[object]] = []
    for r in records:
        prop_rows.append(
            [f"prop_{r.entry_id}_bulk", f"calc_{r.entry_id}",
             "bulk_modulus", r.bulk_modulus, "GPa"]
        )
        prop_rows.append(
            [f"prop_{r.entry_id}_shear", f"calc_{r.entry_id}",
             "shear_modulus", r.shear_modulus, "GPa"]
        )
    _write_csv(
        base_dir / "seed_properties.csv",
        ["property_id", "calculation_id", "property_name", "value", "unit"],
        prop_rows,
    )

    _write_csv(
        base_dir / "seed_prototype_definition.csv",
        ["prototype_id", "prototype_name", "strukturbericht", "formula_type", "description"],
        [
            ["proto_L12", "L12", "L12", "A3B",
             "Cu3Au-type ordered FCC; gamma-prime phase prototype"],
            ["proto_B2", "B2", "B2", "AB",
             "CsCl-type ordered BCC structure"],
            ["proto_A15", "A15", "A15", "A3B",
             "Cr3Si-type cubic structure"],
        ],
    )
    print(f"Generated seed data for {len(records)} compounds in {base_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    generate_seed_files(project_root / "db" / "seed")
