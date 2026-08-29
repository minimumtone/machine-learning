WITH l12_data AS (
    SELECT
        fh.entry_id,
        fh.formula,
        fh.reduced_formula,
        fh.reference_set,
        fh.prototype,
        fh.strukturbericht,
        fh.space_group,
        fh.lattice_a AS lattice_a_angstrom,
        fh.formation_enthalpy_ev_per_atom AS formation_energy_ev_per_atom,
        fh.energy_above_hull,
        fh.is_stable
    FROM formation_enthalpy fh
    WHERE regexp_replace(upper(COALESCE(fh.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
)
SELECT
    entry_id,
    formula,
    reduced_formula,
    reference_set,
    prototype,
    strukturbericht,
    space_group,
    lattice_a_angstrom,
    formation_energy_ev_per_atom,
    energy_above_hull,
    is_stable,
    corr(formation_energy_ev_per_atom, lattice_a_angstrom) OVER () AS pearson_correlation
FROM l12_data
WHERE lattice_a_angstrom IS NOT NULL
  AND formation_energy_ev_per_atom IS NOT NULL
ORDER BY lattice_a_angstrom, formation_energy_ev_per_atom;
