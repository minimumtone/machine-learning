SELECT e.composition_formula
FROM oqmd_entries e
WHERE EXISTS (
    SELECT 1
    FROM oqmd_element_ratios r_ni
    WHERE r_ni.entry_key = e.entry_key
      AND r_ni.symbol = 'Ni'
)
AND EXISTS (
    SELECT 1
    FROM oqmd_element_ratios r_al
    WHERE r_al.entry_key = e.entry_key
      AND r_al.symbol = 'Al'
)
ORDER BY e.composition_formula
LIMIT 10000;
