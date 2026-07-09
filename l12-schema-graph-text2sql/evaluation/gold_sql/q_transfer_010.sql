-- medium: Alを含む安定(凸包上)化合物
SELECT DISTINCT e.composition_formula, f.delta_e
FROM oqmd_entries e
JOIN oqmd_formation_energies f ON f.entry_key = e.entry_key
JOIN oqmd_element_ratios r ON r.entry_key = e.entry_key
WHERE r.symbol = 'Al' AND f.on_hull = true
ORDER BY f.delta_e ASC
LIMIT 10000;
