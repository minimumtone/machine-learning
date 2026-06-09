SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND NOT EXISTS (
    SELECT 1 FROM composition c
    JOIN element e ON e.symbol = c.element
    WHERE c.entry_id = m.entry_id AND e.category NOT IN ('transition_metal')
  )
ORDER BY m.formula
LIMIT 10000;
