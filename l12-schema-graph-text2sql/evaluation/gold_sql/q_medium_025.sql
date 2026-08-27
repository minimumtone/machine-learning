SELECT m.formula, ps.energy_above_hull, ps.is_stable
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Cu'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY ps.energy_above_hull ASC, m.entry_id ASC
LIMIT 10000;