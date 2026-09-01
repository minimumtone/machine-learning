SELECT m.formula, s.prototype, ps.energy_above_hull, ps.is_stable, pde.is_on_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN phase_diagram_entry pde ON pde.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND pde.is_on_hull = TRUE
ORDER BY ps.energy_above_hull ASC, m.formula
LIMIT 10000;
