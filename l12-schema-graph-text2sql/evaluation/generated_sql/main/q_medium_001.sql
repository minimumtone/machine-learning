SELECT m.formula, c.element, c.atomic_fraction, ps.energy_above_hull, ps.is_stable
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition c ON c.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.element = 'Ni'
  AND ps.is_stable = TRUE
ORDER BY m.formula
LIMIT 10000;
