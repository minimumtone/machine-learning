SELECT
  oqmd_element_ratios.symbol,
  AVG(oqmd_formation_energies.hull_distance) AS avg_hull_distance
FROM oqmd_element_ratios
JOIN oqmd_entries
  ON oqmd_element_ratios.entry_key = oqmd_entries.entry_key
JOIN oqmd_formation_energies
  ON oqmd_formation_energies.entry_key = oqmd_entries.entry_key
WHERE oqmd_element_ratios.atomic_ratio = 0.75
GROUP BY oqmd_element_ratios.symbol
ORDER BY AVG(oqmd_formation_energies.hull_distance) ASC
LIMIT 10;
