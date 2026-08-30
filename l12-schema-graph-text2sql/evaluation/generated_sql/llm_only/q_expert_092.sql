SELECT
  s.prototype,
  AVG(et.bulk_modulus_vrh) AS avg_bulk_modulus_vrh
FROM elastic_tensor AS et
JOIN calculation AS c
  ON et.calculation_id = c.calculation_id
JOIN structure AS s
  ON c.entry_id = s.entry_id
WHERE et.bulk_modulus_vrh IS NOT NULL
  AND s.prototype IS NOT NULL
GROUP BY s.prototype
ORDER BY s.prototype;
