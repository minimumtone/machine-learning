SELECT
  method,
  COUNT(DISTINCT entry_id) AS entry_count
FROM calculation
GROUP BY method
ORDER BY entry_count DESC, method;
