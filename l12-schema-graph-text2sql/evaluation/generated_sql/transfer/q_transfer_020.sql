WITH entry_rereferenced AS (
  SELECT
    a.symbol AS a_site,
    e.entry_key,
    fe.delta_e - SUM(er.atomic_ratio * rs.reference_delta_e) AS rereferenced_delta_e
  FROM oqmd_entries e
  JOIN oqmd_formation_energies fe ON fe.entry_key = e.entry_key
  JOIN oqmd_element_ratios a ON a.entry_key = e.entry_key
  JOIN oqmd_element_ratios er ON er.entry_key = e.entry_key
  JOIN oqmd_elements el ON el.symbol = er.symbol
  JOIN oqmd_reference_states rs ON rs.symbol = el.symbol
  WHERE fe.on_hull = TRUE
    AND e.prototype_label IN ('L12', 'L1_2', 'L1₂')
    AND a.wyckoff_site = 'A-site'
  GROUP BY a.symbol, e.entry_key, fe.delta_e
)
SELECT
  a_site_element,
  AVG(rereferenced_delta_e) AS avg_rereferenced_delta_e
FROM entry_rereferenced
GROUP BY a_site_element
ORDER BY a_site_element
LIMIT 10000;
