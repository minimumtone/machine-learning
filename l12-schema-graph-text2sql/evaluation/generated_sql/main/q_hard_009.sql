SELECT
  m.formula,
  s.lattice_a,
  ABS(s.lattice_a - 3.57) AS lattice_a_diff_from_ni3al,
  ps.energy_above_hull,
  ps.is_stable
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND EXISTS (
    SELECT 1
    FROM composition c
    WHERE c.entry_id = m.entry_id
      AND c.element = 'Co'
      AND c.atomic_fraction >= 0.5
  )
ORDER BY ps.energy_above_hull ASC, ABS(s.lattice_a - 3.57) ASC
LIMIT 10000;
