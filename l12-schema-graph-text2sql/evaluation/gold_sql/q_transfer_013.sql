-- hard: Ni-Al両方を含む化合物
SELECT DISTINCT e.entry_key, e.composition_formula
FROM oqmd_entries e
JOIN oqmd_element_ratios ra ON ra.entry_key = e.entry_key AND ra.symbol = 'Ni'
JOIN oqmd_element_ratios rb ON rb.entry_key = e.entry_key AND rb.symbol = 'Al'
LIMIT 10000;
