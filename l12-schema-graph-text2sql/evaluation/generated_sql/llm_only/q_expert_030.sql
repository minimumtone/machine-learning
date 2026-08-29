SELECT
  pd.prototype_id,
  pd.prototype_name,
  COUNT(DISTINCT s.entry_id) AS registration_count
FROM prototype_definition AS pd
LEFT JOIN structure AS s
  ON s.prototype = pd.prototype_id
GROUP BY
  pd.prototype_id,
  pd.prototype_name
ORDER BY
  registration_count DESC,
  pd.prototype_id;
