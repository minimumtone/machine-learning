SELECT
  b.element AS b_site_element,
  AVG(ps.formation_energy_per_atom) AS avg_formation_energy_per_atom
FROM (
  SELECT DISTINCT entry_id, element
  FROM composition
  WHERE site_label = 'B-site'
) AS b
JOIN phase_stability AS ps
  ON ps.entry_id = b.entry_id
WHERE ps.formation_energy_per_atom IS NOT NULL
GROUP BY b.element
ORDER BY b.element;
