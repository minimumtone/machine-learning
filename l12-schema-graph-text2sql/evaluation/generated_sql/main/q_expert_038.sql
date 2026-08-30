SELECT m.formula, m.chemical_system, pde.hull_distance
FROM material_entry m
JOIN phase_diagram_entry pde ON pde.entry_id = m.entry_id
WHERE m.chemical_system = 'Al-Ni'
ORDER BY pde.hull_distance ASC
LIMIT 10000;
