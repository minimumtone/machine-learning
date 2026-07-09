-- hard: A-siteがCoの化合物で形成エネルギーが低い5件
SELECT e.composition_formula, f.delta_e
FROM oqmd_entries e
JOIN oqmd_element_ratios r ON r.entry_key = e.entry_key
JOIN oqmd_formation_energies f ON f.entry_key = e.entry_key
WHERE r.symbol = 'Co' AND r.wyckoff_site = 'A-site'
ORDER BY f.delta_e ASC
LIMIT 5;
