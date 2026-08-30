SELECT
  COUNT(DISTINCT m.entry_id) FILTER (WHERE ps.is_stable = TRUE) AS stable_l12_count,
  COUNT(DISTINCT m.entry_id) FILTER (WHERE ps.is_stable = FALSE) AS not_stable_l12_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition c ON c.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.element = 'Al'
LIMIT 10000;
