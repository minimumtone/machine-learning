SELECT
  c.element AS b_site_element,
  AVG(fh.enthalpy_vs_element_ground_states) AS avg_rereferenced_formation_energy_ev_per_atom,
  COUNT(*) AS num_entries
FROM composition c
JOIN formation_enthalpy fh
  ON fh.entry_id = c.entry_id
WHERE c.site_label = 'B-site'
  AND fh.enthalpy_vs_element_ground_states IS NOT NULL
GROUP BY c.element
HAVING AVG(fh.enthalpy_vs_element_ground_states) < -0.4
ORDER BY avg_rereferenced_formation_energy_ev_per_atom;
