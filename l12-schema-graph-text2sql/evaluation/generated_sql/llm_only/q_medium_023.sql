SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.lattice_a AS lattice_constant_angstrom
FROM structure AS s
JOIN material_entry AS me
    ON me.entry_id = s.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE s.lattice_a >= 4.0
  AND (
      regexp_replace(upper(translate(COALESCE(s.strukturbericht, ''), '₂', '2')), '[^A-Z0-9]', '', 'g') = 'L12'
      OR regexp_replace(upper(translate(COALESCE(pd.strukturbericht, ''), '₂', '2')), '[^A-Z0-9]', '', 'g') = 'L12'
  )
ORDER BY s.lattice_a, me.formula;
