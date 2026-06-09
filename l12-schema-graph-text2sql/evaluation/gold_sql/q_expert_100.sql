SELECT s.prototype,
  SUM(CASE WHEN ps.is_stable = TRUE THEN 1 ELSE 0 END) AS stable,
  SUM(CASE WHEN ps.is_stable = FALSE AND ps.energy_above_hull < 0.1 THEN 1 ELSE 0 END) AS metastable,
  SUM(CASE WHEN ps.is_stable = FALSE AND ps.energy_above_hull >= 0.1 THEN 1 ELSE 0 END) AS unstable,
  COUNT(*) AS total
FROM structure s
JOIN phase_stability ps ON ps.entry_id = s.entry_id
GROUP BY s.prototype
ORDER BY total DESC
LIMIT 10000;
