SELECT m.formula, s.prototype
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'NaCl' OR s.strukturbericht = 'B1')
  AND EXISTS (
    SELECT 1
    FROM composition c
    WHERE c.entry_id = m.entry_id
      AND c.element = 'Sc'
  )
ORDER BY m.formula
LIMIT 10000;
