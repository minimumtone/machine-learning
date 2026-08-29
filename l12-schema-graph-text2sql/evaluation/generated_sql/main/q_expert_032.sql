SELECT m.formula, ps.formation_energy_per_atom
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE ps.formation_energy_per_atom > 0
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;
