SELECT COUNT(DISTINCT s.entry_id) AS count
FROM structure s
LEFT JOIN prototype_definition pd
  ON s.prototype = pd.prototype_id
WHERE s.strukturbericht = 'B1'
   OR pd.strukturbericht = 'B1'
   OR s.prototype ILIKE '%NaCl%'
   OR pd.prototype_name ILIKE '%NaCl%';
