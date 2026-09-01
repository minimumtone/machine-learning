SELECT
  c.element AS b_site_element,
  COUNT(DISTINCT c.entry_id) AS stable_l12_compound_count
FROM composition c
JOIN structure s
  ON s.entry_id = c.entry_id
JOIN phase_stability ps
  ON ps.entry_id = c.entry_id
WHERE c.site_label = 'B-site'
  AND ps.is_stable = TRUE
  AND s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
GROUP BY c.element
ORDER BY stable_l12_compound_count DESC, b_site_element;
