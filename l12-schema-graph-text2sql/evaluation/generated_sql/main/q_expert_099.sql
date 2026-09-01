SELECT
  FLOOR(s.lattice_a * 10.0) / 10.0 AS lattice_a_bin_start,
  FLOOR(s.lattice_a * 10.0) / 10.0 + 0.1 AS lattice_a_bin_end,
  COUNT(*) AS lattice_a_bin_count
FROM structure s
WHERE s.lattice_a IS NOT NULL
GROUP BY FLOOR(s.lattice_a * 10.0) / 10.0
ORDER BY lattice_a_bin_start ASC
LIMIT 10000;
