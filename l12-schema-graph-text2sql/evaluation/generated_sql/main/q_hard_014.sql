SELECT DISTINCT
    m.entry_id, m.formula, s.prototype, s.lattice_a, s.space_group, ps.formation_energy_per_atom, ps.energy_above_hull, ps.band_gap
FROM material_entry m
    JOIN composition c ON c.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN structure s ON s.entry_id = m.entry_id
WHERE
    (s.prototype = 'L12' OR s.strukturbericht = 'L12')
    AND ps.is_stable = TRUE
    AND c.site_label = 'A-site'

LIMIT 10000;
