SELECT pde.chemical_system, m.formula, pde.hull_distance
FROM phase_diagram_entry pde
JOIN material_entry m ON m.entry_id = pde.entry_id
WHERE pde.is_on_hull = TRUE
ORDER BY pde.chemical_system, pde.hull_distance ASC, m.formula
LIMIT 10000;
