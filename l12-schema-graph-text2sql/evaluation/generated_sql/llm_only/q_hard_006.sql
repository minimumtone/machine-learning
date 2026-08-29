SELECT
  c.element AS a_site_element,
  AVG(s.lattice_a) AS avg_lattice_a,
  COUNT(DISTINCT s.entry_id) AS num_compounds
FROM structure s
JOIN composition c
  ON c.entry_id = s.entry_id
LEFT JOIN prototype_definition pd
  ON pd.prototype_id = s.prototype
WHERE c.site_label = 'A-site'
  AND s.lattice_a IS NOT NULL
  AND UPPER(COALESCE(s.strukturbericht, pd.strukturbericht)) IN ('L1_2', 'L12', 'L1₂')
GROUP BY c.element
ORDER BY c.element;
