SELECT DISTINCT
    m.entry_id, m.formula, s.prototype, s.lattice_a, s.space_group, ps.formation_energy_per_atom, ps.energy_above_hull, ps.band_gap
FROM material_entry m
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN structure s ON s.entry_id = m.entry_id
WHERE
    (s.prototype = 'NiAs' OR s.strukturbericht = 'NiAs')
    AND EXISTS (SELECT 1 FROM composition c_as WHERE c_as.entry_id = m.entry_id AND c_as.element = 'As')
    AND EXISTS (SELECT 1 FROM composition c_ni WHERE c_ni.entry_id = m.entry_id AND c_ni.element = 'Ni')
    AND ps.is_stable = TRUE
    AND EXISTS (SELECT 1 FROM composition c_sl WHERE c_sl.entry_id = m.entry_id AND c_sl.site_label IN ('A-site', 'B-site'))

LIMIT 10000;
