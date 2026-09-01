SELECT
  e.composition_formula,
  fe.delta_e,
  fe.hull_distance,
  fe.on_hull
FROM oqmd_entries e
JOIN oqmd_formation_energies fe
  ON fe.entry_key = e.entry_key
WHERE fe.on_hull = TRUE
  AND EXISTS (
    SELECT 1
    FROM oqmd_element_ratios er
    WHERE er.entry_key = e.entry_key
      AND er.symbol = 'Al'
  )
ORDER BY fe.delta_e ASC
LIMIT 10000;
