SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN ps.energy_above_hull > 0.001 THEN 1 ELSE 0 END) AS unstable,
  ROUND(100.0 * SUM(CASE WHEN ps.energy_above_hull > 0.001 THEN 1 ELSE 0 END) / COUNT(*), 2) AS unstable_pct
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'NaCl' OR s.strukturbericht = 'B1'
LIMIT 10000;
