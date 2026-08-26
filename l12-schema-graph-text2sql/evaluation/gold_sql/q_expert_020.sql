SELECT m.entry_id, m.formula, ps.energy_above_hull
FROM material_entry m
JOIN composition c_v ON c_v.entry_id = m.entry_id AND c_v.element = 'V'
JOIN composition c_al ON c_al.entry_id = m.entry_id AND c_al.element = 'Al'
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE ps.energy_above_hull <= 0.001
ORDER BY m.formula
LIMIT 10000;
