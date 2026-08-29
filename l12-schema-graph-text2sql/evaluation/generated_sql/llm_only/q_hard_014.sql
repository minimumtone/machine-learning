SELECT
  c.element AS a_site_element,
  AVG(fh.formation_enthalpy_ev_per_atom) AS avg_formation_energy_ev_per_atom
FROM formation_enthalpy fh
JOIN composition c
  ON c.entry_id = fh.entry_id
WHERE fh.is_stable = TRUE
  AND fh.strukturbericht = 'L1_2'
  AND c.site_label = 'A-site'
GROUP BY c.element
ORDER BY c.element;
