SELECT me.formula, me.entry_id, s.volume_per_atom
FROM structure s
JOIN material_entry me ON me.entry_id = s.entry_id
LEFT JOIN prototype_definition pd ON pd.prototype_id = s.prototype
WHERE s.volume_per_atom IS NOT NULL
  AND (
    UPPER(REPLACE(s.strukturbericht, '_', '')) = 'L12'
    OR UPPER(REPLACE(pd.strukturbericht, '_', '')) = 'L12'
  )
ORDER BY s.volume_per_atom ASC
LIMIT 1;
