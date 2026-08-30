SELECT
  s.prototype,
  pd.prototype_name,
  COUNT(DISTINCT me.entry_id) AS mn_compound_count
FROM material_entry AS me
JOIN composition AS c
  ON c.entry_id = me.entry_id
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE c.element = 'Mn'
  AND me.number_of_elements > 1
  AND s.prototype IS NOT NULL
GROUP BY s.prototype, pd.prototype_name
ORDER BY mn_compound_count DESC, s.prototype;
