SELECT
  COUNT(*) FILTER (WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12') AND ps.is_stable = TRUE) * 100.0
    / NULLIF(COUNT(*) FILTER (WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')), 0) AS l12_stable_percentage,
  COUNT(*) FILTER (WHERE (s.prototype = 'B2' OR s.strukturbericht = 'B2') AND ps.is_stable = TRUE) * 100.0
    / NULLIF(COUNT(*) FILTER (WHERE (s.prototype = 'B2' OR s.strukturbericht = 'B2')), 0) AS b2_stable_percentage
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'L12'
   OR s.strukturbericht = 'L12'
   OR s.prototype = 'B2'
   OR s.strukturbericht = 'B2'
LIMIT 10000;
