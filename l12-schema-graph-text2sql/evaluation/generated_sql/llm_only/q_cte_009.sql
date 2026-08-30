SELECT
  c.element,
  COUNT(DISTINCT fh.entry_id) AS stable_l12_compound_count,
  AVG(fh.enthalpy_vs_element_ground_states) AS avg_formation_energy_vs_element_ground_states
FROM formation_enthalpy fh
JOIN composition c
  ON c.entry_id = fh.entry_id
WHERE fh.is_stable = TRUE
  AND fh.strukturbericht = 'L1_2'
  AND fh.enthalpy_vs_element_ground_states IS NOT NULL
GROUP BY c.element
ORDER BY stable_l12_compound_count DESC, c.element
LIMIT 5;
