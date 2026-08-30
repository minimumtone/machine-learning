SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system
FROM material_entry AS me
LEFT JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
LEFT JOIN formation_enthalpy AS fh
  ON fh.entry_id = me.entry_id
WHERE me.chemical_system <> 'Al-Ni'
  AND (
    UPPER(REPLACE(REPLACE(COALESCE(s.strukturbericht, ''), '_', ''), '₂', '2')) = 'L12'
    OR UPPER(REPLACE(REPLACE(COALESCE(pd.strukturbericht, ''), '_', ''), '₂', '2')) = 'L12'
    OR UPPER(REPLACE(REPLACE(COALESCE(fh.strukturbericht, ''), '_', ''), '₂', '2')) = 'L12'
  )
ORDER BY me.chemical_system, me.reduced_formula, me.entry_id;
