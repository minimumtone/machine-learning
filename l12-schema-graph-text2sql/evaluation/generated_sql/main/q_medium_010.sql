SELECT DISTINCT m.formula, s.prototype, s.strukturbericht
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND (
    EXISTS (
      SELECT 1
      FROM composition c_ni
      WHERE c_ni.entry_id = m.entry_id
        AND c_ni.element = 'Ni'
    )
    OR EXISTS (
      SELECT 1
      FROM composition c_co
      WHERE c_co.entry_id = m.entry_id
        AND c_co.element = 'Co'
    )
  )
ORDER BY m.formula
LIMIT 10000;
