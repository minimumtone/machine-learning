SELECT m.formula, ps.formation_energy_per_atom, ps.energy_above_hull, ps.is_stable
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND EXISTS (
      SELECT 1
      FROM composition c
      WHERE c.entry_id = m.entry_id
        AND c.element IN ('Ni', 'Co')
  )
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;
