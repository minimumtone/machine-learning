SELECT
    m.formula,
    AVG(pde.hull_distance) AS avg_energy_above_hull
FROM material_entry m
JOIN phase_diagram_entry pde ON pde.entry_id = m.entry_id
GROUP BY m.formula
ORDER BY avg_energy_above_hull ASC
LIMIT 10000;
