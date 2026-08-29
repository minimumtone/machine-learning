SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  MAX(se.work_function) AS max_work_function_ev
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
JOIN surface_energy AS se
  ON se.entry_id = me.entry_id
WHERE se.work_function >= 5.0
  AND (
    UPPER(REPLACE(COALESCE(s.strukturbericht, ''), '_', '')) = 'L12'
    OR UPPER(REPLACE(COALESCE(pd.strukturbericht, ''), '_', '')) = 'L12'
    OR UPPER(REPLACE(COALESCE(pd.prototype_name, ''), '_', '')) LIKE '%L12%'
  )
GROUP BY
  me.entry_id,
  me.formula,
  me.reduced_formula
ORDER BY
  max_work_function_ev DESC;
