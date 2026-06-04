SELECT
  FLOOR(lattice_a * 10) / 10.0 AS bin_start,
  COUNT(*) AS cnt
FROM structure
WHERE lattice_a IS NOT NULL
GROUP BY FLOOR(lattice_a * 10)
ORDER BY bin_start
LIMIT 10000;
