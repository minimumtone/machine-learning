SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN NOT ps.is_stable THEN 1 ELSE 0 END) AS not_stable,
  ROUND(100.0 * SUM(CASE WHEN NOT ps.is_stable THEN 1 ELSE 0 END) / COUNT(*), 2) AS not_stable_pct
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'NaCl' OR s.strukturbericht = 'B1'
LIMIT 10000;
