SELECT
  prototype,
  COUNT(*) AS num_entries,
  AVG(energy_above_hull) AS avg_energy_above_hull_ev_per_atom
FROM formation_enthalpy
WHERE prototype IS NOT NULL
  AND energy_above_hull IS NOT NULL
GROUP BY prototype
ORDER BY avg_energy_above_hull_ev_per_atom ASC;
