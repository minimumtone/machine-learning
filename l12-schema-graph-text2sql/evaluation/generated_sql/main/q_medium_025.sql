SELECT
  m.formula,
  CASE
    WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
    WHEN ps.energy_above_hull <= 0.05 THEN 'metastable'
    ELSE 'unstable'
  END AS stability,
  ps.energy_above_hull,
  ps.is_stable
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Cu'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY
  CASE
    WHEN ps.energy_above_hull <= 0.001 THEN 1
    WHEN ps.energy_above_hull <= 0.05 THEN 2
    ELSE 3
  END,
  ps.energy_above_hull ASC,
  m.formula
LIMIT 10000;
