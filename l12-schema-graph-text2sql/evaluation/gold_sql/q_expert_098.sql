SELECT
  CASE WHEN ps.band_gap = 0 OR ps.band_gap IS NULL THEN 'metallic (gap=0)' ELSE 'non-zero gap' END AS gap_category,
  COUNT(*) AS cnt,
  AVG(ps.band_gap) AS avg_gap
FROM structure s
JOIN phase_stability ps ON ps.entry_id = s.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
GROUP BY gap_category
LIMIT 10000;
