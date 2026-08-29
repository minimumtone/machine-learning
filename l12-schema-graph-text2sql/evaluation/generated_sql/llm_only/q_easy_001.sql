SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
   OR pd.strukturbericht IN ('L1_2', 'L12', 'L1₂')
ORDER BY me.chemical_system, me.reduced_formula;
