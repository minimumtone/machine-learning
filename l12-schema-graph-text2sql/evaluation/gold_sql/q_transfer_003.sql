-- easy: Niを含むエントリ
SELECT DISTINCT e.entry_key, e.composition_formula
FROM oqmd_entries e
JOIN oqmd_element_ratios r ON r.entry_key = e.entry_key
WHERE r.symbol = 'Ni'
LIMIT 10000;
