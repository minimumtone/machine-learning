SELECT DISTINCT
    m.entry_id, m.formula, s.prototype, s.lattice_a, s.space_group, ps.formation_energy_per_atom, ps.energy_above_hull, ps.band_gap
FROM material_entry m
    JOIN composition c ON c.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN structure s ON s.entry_id = m.entry_id
WHERE
    (s.prototype = 'BiF3' OR s.strukturbericht = 'BiF3')
    AND ps.is_stable = TRUE
    AND (m.formula = 'BiF3' OR m.reduced_formula = 'BiF3')
    AND c.site_label IN ('A-site', 'B-site')

LIMIT 10000;
