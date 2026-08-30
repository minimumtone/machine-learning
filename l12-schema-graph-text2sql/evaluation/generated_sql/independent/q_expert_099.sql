SELECT
  bins.lattice_a_bin_start,
  bins.lattice_a_bin_start + 0.1 AS lattice_a_bin_end,
  COUNT(*) AS lattice_a_bin_count
FROM (
  SELECT FLOOR(s.lattice_a * 10) / 10.0 AS lattice_a_bin_start
  FROM structure s
  WHERE s.lattice_a IS NOT NULL
) bins
GROUP BY bins.lattice_a_bin_start
ORDER BY bins.lattice_a_bin_start ASC
LIMIT 10000;
