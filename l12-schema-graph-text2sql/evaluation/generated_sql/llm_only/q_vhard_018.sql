SELECT
  c.element AS a_site_element,
  AVG(fh.enthalpy_vs_element_ground_states) AS avg_rebased_formation_energy_ev_per_atom
FROM formation_enthalpy AS fh
JOIN composition AS c
  ON c.entry_id = fh.entry_id
WHERE fh.is_stable = TRUE
  AND c.site_label = 'A-site'
  AND (
    fh.strukturbericht = 'L1₂'
    OR fh.strukturbericht = 'L12'
    OR fh.prototype ILIKE '%L1_2%'
    OR fh.prototype ILIKE '%L12%'
  )
GROUP BY c.element
ORDER BY c.element;
