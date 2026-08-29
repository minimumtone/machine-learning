SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.prototype,
    s.strukturbericht,
    s.space_group,
    et.poisson_ratio,
    et.bulk_modulus_vrh / NULLIF(et.shear_modulus_vrh, 0) AS pugh_ratio
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
WHERE regexp_replace(upper(COALESCE(s.strukturbericht, pd.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
  AND et.poisson_ratio < 0.25
  AND et.bulk_modulus_vrh / NULLIF(et.shear_modulus_vrh, 0) < 1.75
ORDER BY et.poisson_ratio ASC;
