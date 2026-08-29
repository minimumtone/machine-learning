SELECT e.composition_formula
FROM oqmd_entries e
WHERE EXISTS (
  SELECT 1
  FROM oqmd_element_ratios r
  WHERE r.entry_key = e.entry_key
    AND r.symbol = 'Ni'
)
ORDER BY e.composition_formula
LIMIT 10000;
