SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system,
  MIN(se.surface_energy_j_m2) AS min_surface_energy_j_m2
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
JOIN surface_energy AS se
  ON se.entry_id = me.entry_id
WHERE ps.is_stable = TRUE
  AND se.surface_energy_j_m2 < 1.5
  AND REPLACE(UPPER(s.strukturbericht), '_', '') = 'L12'
GROUP BY
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system
ORDER BY
  min_surface_energy_j_m2 ASC;
