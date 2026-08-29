WITH binned AS (
  SELECT
    (FLOOR(lattice_a * 10) / 10.0) AS bin_start
  FROM structure
  WHERE lattice_a IS NOT NULL
)
SELECT
  bin_start,
  bin_start + 0.1 AS bin_end,
  COUNT(*) AS count
FROM binned
GROUP BY bin_start
ORDER BY bin_start;
