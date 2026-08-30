SELECT
  oqmd_entries.composition_formula,
  oqmd_element_ratios.symbol,
  oqmd_element_ratios.wyckoff_site,
  oqmd_formation_energies.delta_e
FROM oqmd_element_ratios
JOIN oqmd_entries
  ON oqmd_element_ratios.entry_key = oqmd_entries.entry_key
JOIN oqmd_formation_energies
  ON oqmd_formation_energies.entry_key = oqmd_entries.entry_key
WHERE oqmd_element_ratios.symbol = 'Co'
  AND oqmd_element_ratios.wyckoff_site = 'A-site'
ORDER BY oqmd_formation_energies.delta_e ASC
LIMIT 5;
