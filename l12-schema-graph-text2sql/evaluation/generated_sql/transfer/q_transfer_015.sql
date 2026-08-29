SELECT
  oqmd_entries.composition_formula,
  oqmd_entries.spacegroup_number,
  oqmd_formation_energies.gap_ev
FROM oqmd_entries
JOIN oqmd_formation_energies
  ON oqmd_formation_energies.entry_key = oqmd_entries.entry_key
WHERE oqmd_entries.spacegroup_number = 221
  AND oqmd_formation_energies.gap_ev > 0
ORDER BY oqmd_formation_energies.gap_ev DESC
LIMIT 10000;
