SELECT s.prototype,
  COUNT(*) AS total,
  SUM(CASE WHEN ps.is_stable = TRUE THEN 1 ELSE 0 END) AS stable,
  ROUND(100.0 * SUM(CASE WHEN ps.is_stable = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS stable_pct
FROM structure s
JOIN phase_stability ps ON ps.entry_id = s.entry_id
WHERE s.prototype IN ('L12', 'B2')
GROUP BY s.prototype
ORDER BY s.prototype
LIMIT 10000;
