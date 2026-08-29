SELECT DISTINCT
    m.entry_id, m.formula, ps.formation_energy_per_atom, ps.energy_above_hull, ps.band_gap
FROM material_entry m
    JOIN composition c ON c.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE
    c.site_label = 'B-site'

LIMIT 10000;
