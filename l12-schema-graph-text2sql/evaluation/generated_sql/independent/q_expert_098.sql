SELECT m.formula,
       ps.band_gap,
       CASE
         WHEN ps.band_gap = 0 THEN 'zero_band_gap'
         ELSE 'nonzero_band_gap'
       END AS band_gap_category
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.band_gap IS NOT NULL
ORDER BY band_gap_category, ps.band_gap ASC, m.formula
LIMIT 10000;
