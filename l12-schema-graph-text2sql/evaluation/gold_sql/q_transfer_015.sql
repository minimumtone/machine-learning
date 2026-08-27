-- hard: 空間群221でバンドギャップ>0の化合物
SELECT e.composition_formula, f.gap_ev
FROM oqmd_entries e
JOIN oqmd_formation_energies f ON f.entry_key = e.entry_key
WHERE e.spacegroup_number = 221 AND f.gap_ev > 0
ORDER BY f.gap_ev DESC, e.entry_key ASC
LIMIT 10000;
