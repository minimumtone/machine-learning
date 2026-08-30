SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    s.prototype,
    s.strukturbericht
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
WHERE (
    upper(regexp_replace(coalesce(s.strukturbericht, ''), '[^A-Za-z0-9]', '', 'g')) = 'L12'
    OR upper(regexp_replace(coalesce(s.prototype, ''), '[^A-Za-z0-9]', '', 'g')) = 'L12'
)
AND me.number_of_elements > 1
AND NOT EXISTS (
    SELECT 1
    FROM unnest(string_to_array(me.chemical_system, '-')) AS cs(element_symbol)
    JOIN element AS e
      ON e.symbol = cs.element_symbol
    WHERE e.category <> 'transition_metal'
);
