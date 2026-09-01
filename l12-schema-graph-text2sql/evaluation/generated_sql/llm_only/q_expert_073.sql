SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  ms.temperature_k
FROM material_entry AS me
JOIN material_synthesis AS ms
  ON ms.entry_id = me.entry_id
LEFT JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
LEFT JOIN formation_enthalpy AS fh
  ON fh.entry_id = me.entry_id
WHERE ms.temperature_k >= 1000
  AND (
    regexp_replace(upper(COALESCE(s.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(COALESCE(pd.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(COALESCE(fh.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(COALESCE(s.prototype, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(COALESCE(pd.prototype_name, '')), '[^A-Z0-9]', '', 'g') = 'L12'
  )
ORDER BY me.formula, ms.temperature_k;
