SELECT
  c.element AS a_site_element,
  COUNT(DISTINCT c.entry_id) AS l12_compound_count
FROM composition c
JOIN structure s
  ON s.entry_id = c.entry_id
LEFT JOIN prototype_definition pd
  ON pd.prototype_id = s.prototype
WHERE c.site_label = 'A-site'
  AND (
    s.strukturbericht IN ('L1_2', 'L1₂', 'L12')
    OR pd.strukturbericht IN ('L1_2', 'L1₂', 'L12')
    OR s.prototype ILIKE '%L1_2%'
    OR s.prototype ILIKE '%L1₂%'
    OR s.prototype ILIKE '%L12%'
    OR pd.prototype_name ILIKE '%L1_2%'
    OR pd.prototype_name ILIKE '%L1₂%'
    OR pd.prototype_name ILIKE '%L12%'
  )
GROUP BY c.element
ORDER BY l12_compound_count DESC, c.element;
