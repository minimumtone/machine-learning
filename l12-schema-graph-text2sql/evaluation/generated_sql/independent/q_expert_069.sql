SELECT m.formula, se.miller_index, se.surface_energy_j_m2
FROM material_entry m
JOIN surface_energy se ON se.entry_id = m.entry_id
WHERE se.miller_index IN ('100', '110')
  AND EXISTS (
    SELECT 1
    FROM surface_energy se100
    WHERE se100.entry_id = m.entry_id
      AND se100.miller_index = '100'
  )
  AND EXISTS (
    SELECT 1
    FROM surface_energy se110
    WHERE se110.entry_id = m.entry_id
      AND se110.miller_index = '110'
  )
ORDER BY m.formula, se.miller_index
LIMIT 10000;
