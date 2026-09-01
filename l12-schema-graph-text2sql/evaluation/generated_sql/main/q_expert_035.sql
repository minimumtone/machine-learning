SELECT
  COUNT(*) FILTER (WHERE ps.is_stable = FALSE) * 100.0 / NULLIF(COUNT(*), 0) AS nacl_unstable_percentage
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'NaCl' OR s.strukturbericht = 'B1')
LIMIT 10000;
