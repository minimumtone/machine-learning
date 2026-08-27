SELECT m.entry_id, m.formula, ps.energy_above_hull, ps.is_stable
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE m.chemical_system = 'Al-Ni'
ORDER BY ps.energy_above_hull
LIMIT 10000;
