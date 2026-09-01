SELECT
  symbol,
  COUNT(DISTINCT entry_key) AS entry_count
FROM oqmd_element_ratios
GROUP BY symbol
ORDER BY entry_count DESC
LIMIT 10;
