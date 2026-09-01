SELECT m.formula, ps.energy_above_hull, ps.is_stable
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE ps.is_stable = TRUE
  AND EXISTS (
    SELECT 1
    FROM composition c_v
    WHERE c_v.entry_id = m.entry_id
      AND c_v.element = 'V'
  )
  AND EXISTS (
    SELECT 1
    FROM composition c_al
    WHERE c_al.entry_id = m.entry_id
      AND c_al.element = 'Al'
  )
ORDER BY ps.energy_above_hull ASC
LIMIT 10000;
