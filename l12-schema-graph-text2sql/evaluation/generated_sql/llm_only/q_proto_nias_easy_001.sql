SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system
FROM material_entry AS me
WHERE EXISTS (
  SELECT 1
  FROM structure AS s
  LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
  WHERE s.entry_id = me.entry_id
    AND (
      s.prototype ILIKE '%NiAs%'
      OR s.strukturbericht IN ('B8_1', 'B81')
      OR pd.prototype_name ILIKE '%NiAs%'
      OR pd.description ILIKE '%NiAs%'
    )
)
OR EXISTS (
  SELECT 1
  FROM formation_enthalpy AS fh
  WHERE fh.entry_id = me.entry_id
    AND (
      fh.prototype ILIKE '%NiAs%'
      OR fh.strukturbericht IN ('B8_1', 'B81')
    )
)
ORDER BY me.reduced_formula, me.entry_id;
