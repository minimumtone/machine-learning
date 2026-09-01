SELECT
  oqmd_entries.crystal_system,
  AVG(oqmd_formation_energies.delta_e) AS avg_eform
FROM oqmd_entries
JOIN oqmd_formation_energies
  ON oqmd_formation_energies.entry_key = oqmd_entries.entry_key
GROUP BY oqmd_entries.crystal_system
ORDER BY avg_formation_energy ASC
LIMIT 10000;
