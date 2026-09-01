SELECT m.formula, ps.energy_above_hull, ps.is_stable
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND EXISTS (
    SELECT 1
    FROM composition c
    WHERE c.entry_id = m.entry_id
      AND c.element = 'Ni'
  )
ORDER BY ps.energy_above_hull ASC, m.formula
LIMIT 10000;
