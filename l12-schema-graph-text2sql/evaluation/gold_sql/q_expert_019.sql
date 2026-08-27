-- Resolves the L12 prototype through the prototype_definition master so the
-- benchmark exercises the controlled-vocabulary relation.
SELECT m.entry_id, m.formula, c.element, e.atomic_number
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN prototype_definition pd ON pd.prototype_id = s.prototype
JOIN element e ON e.symbol = c.element
WHERE e.atomic_number >= 40
  AND (pd.prototype_id = 'L12' OR pd.strukturbericht = 'L12')
ORDER BY m.formula, m.entry_id, c.element
LIMIT 10000;
