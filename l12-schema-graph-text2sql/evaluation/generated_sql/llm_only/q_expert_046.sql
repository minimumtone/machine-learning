SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    et.bulk_modulus_vrh,
    et.shear_modulus_vrh,
    et.bulk_modulus_vrh / et.shear_modulus_vrh AS bg_ratio
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
WHERE et.shear_modulus_vrh > 0
  AND et.bulk_modulus_vrh / et.shear_modulus_vrh >= 2
  AND (
      regexp_replace(upper(COALESCE(s.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
      OR regexp_replace(upper(COALESCE(pd.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
      OR regexp_replace(upper(COALESCE(s.prototype, '')), '[^A-Z0-9]', '', 'g') = 'L12'
      OR regexp_replace(upper(COALESCE(pd.prototype_name, '')), '[^A-Z0-9]', '', 'g') = 'L12'
  )
ORDER BY bg_ratio DESC;
