SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    c.element AS b_site_element,
    s.prototype,
    COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
JOIN composition AS c
    ON c.entry_id = me.entry_id
JOIN element AS e
    ON e.symbol = c.element
WHERE c.site_label = 'B-site'
  AND e.category = 'transition_metal'
  AND e.period_number = 5
  AND e.block = 'd'
  AND (
      UPPER(REPLACE(COALESCE(s.strukturbericht, pd.strukturbericht, ''), '_', '')) = 'L12'
      OR UPPER(REPLACE(COALESCE(s.prototype, pd.prototype_name, ''), '_', '')) LIKE '%L12%'
  )
ORDER BY me.reduced_formula, c.element;
