SELECT
  CASE WHEN ps.energy_above_hull <= 0.001 THEN 'stable' ELSE 'not_stable' END AS stability,
  COUNT(*) AS count
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Al'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
GROUP BY CASE WHEN ps.energy_above_hull <= 0.001 THEN 'stable' ELSE 'not_stable' END
ORDER BY stability
LIMIT 10000;