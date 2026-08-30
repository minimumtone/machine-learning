SELECT DISTINCT
    m.entry_id, m.formula, s.prototype, s.lattice_a, s.space_group, ps.formation_energy_per_atom, ps.energy_above_hull, ps.band_gap
FROM material_entry m
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN structure s ON s.entry_id = m.entry_id
WHERE
    (s.prototype = 'L12' OR s.strukturbericht = 'L12')
    AND ps.formation_energy_per_atom < 0

LIMIT 10000;
