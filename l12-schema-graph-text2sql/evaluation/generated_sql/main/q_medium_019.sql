SELECT m.formula, ps.energy_above_hull, ps.is_stable
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
ORDER BY ps.energy_above_hull ASC
LIMIT 10000;
