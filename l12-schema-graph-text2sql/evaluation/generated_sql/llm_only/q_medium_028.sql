SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    s.prototype,
    COALESCE(s.strukturbericht, pd.strukturbericht, fh.strukturbericht) AS strukturbericht,
    s.space_group_number,
    s.space_group
FROM material_entry AS me
JOIN composition AS c
    ON c.entry_id = me.entry_id
LEFT JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
LEFT JOIN formation_enthalpy AS fh
    ON fh.entry_id = me.entry_id
WHERE c.element = 'Ir'
  AND me.number_of_elements > 1
  AND (
      regexp_replace(upper(translate(COALESCE(s.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'
      OR regexp_replace(upper(translate(COALESCE(pd.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'
      OR regexp_replace(upper(translate(COALESCE(fh.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'
      OR regexp_replace(upper(translate(COALESCE(pd.prototype_name, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
  )
ORDER BY me.reduced_formula, me.entry_id;
