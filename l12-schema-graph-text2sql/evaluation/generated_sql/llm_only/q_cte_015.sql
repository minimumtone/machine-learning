SELECT
  c.element AS a_site_element,
  AVG(fh.enthalpy_vs_element_ground_states) AS avg_referenced_formation_energy_ev_per_atom
FROM formation_enthalpy fh
JOIN composition c
  ON c.entry_id = fh.entry_id
JOIN pure_element_reference per
  ON per.element_symbol = c.element
 AND per.reference_set = fh.reference_set
WHERE fh.is_stable = TRUE
  AND c.site_label = 'A-site'
  AND per.volume_per_atom >= 15
  AND fh.strukturbericht IN ('L1_2', 'L12', 'L1₂')
  AND fh.enthalpy_vs_element_ground_states IS NOT NULL
GROUP BY c.element
ORDER BY c.element;
