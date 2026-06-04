SELECT m.entry_id, m.formula, ps.band_gap
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND (ps.band_gap = 0 OR ps.band_gap IS NULL)
ORDER BY m.formula
LIMIT 10000;
