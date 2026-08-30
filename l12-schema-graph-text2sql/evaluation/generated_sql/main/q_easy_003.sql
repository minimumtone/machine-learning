SELECT DISTINCT m.formula
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
WHERE m.number_of_elements = 2
  AND EXISTS (
    SELECT 1
    FROM composition c_a
    WHERE c_a.entry_id = m.entry_id
      AND c_a.atomic_fraction = 0.75
  )
  AND EXISTS (
    SELECT 1
    FROM composition c_b
    WHERE c_b.entry_id = m.entry_id
      AND c_b.atomic_fraction = 0.25
  )
ORDER BY m.formula
LIMIT 10000;
