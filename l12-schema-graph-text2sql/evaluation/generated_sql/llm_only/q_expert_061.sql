SELECT
  me.entry_id,
  me.formula,
  se.surface_energy_j_m2
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
JOIN surface_energy AS se
  ON se.entry_id = me.entry_id
WHERE regexp_replace(se.miller_index, '[^0-9-]', '', 'g') = '111'
  AND (
    upper(replace(COALESCE(s.strukturbericht, ''), '_', '')) = 'L12'
    OR upper(replace(COALESCE(pd.strukturbericht, ''), '_', '')) = 'L12'
    OR s.prototype ILIKE '%L1_2%'
    OR s.prototype ILIKE '%L12%'
    OR pd.prototype_name ILIKE '%L1_2%'
    OR pd.prototype_name ILIKE '%L12%'
  )
ORDER BY se.surface_energy_j_m2 ASC
LIMIT 1;
