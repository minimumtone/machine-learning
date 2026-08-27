SELECT m.entry_id, m.formula, ps.band_gap
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.band_gap > 0
ORDER BY ps.band_gap DESC, m.entry_id ASC
LIMIT 10000;
