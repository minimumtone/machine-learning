SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system,
  s.prototype,
  COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht,
  s.space_group_number,
  COALESCE(s.space_group, sg.hermann_mauguin) AS space_group
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
LEFT JOIN space_group AS sg
  ON sg.space_group_number = s.space_group_number
WHERE
  regexp_replace(upper(translate(COALESCE(s.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'
  OR regexp_replace(upper(translate(COALESCE(pd.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'
  OR regexp_replace(upper(translate(COALESCE(s.prototype, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'
  OR regexp_replace(upper(translate(COALESCE(pd.prototype_id, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'
  OR regexp_replace(upper(translate(COALESCE(pd.prototype_name, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
ORDER BY me.reduced_formula, me.entry_id;
