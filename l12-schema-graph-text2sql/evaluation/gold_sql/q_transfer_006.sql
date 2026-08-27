-- easy: バンドギャップ最大5件
SELECT e.composition_formula, f.gap_ev
FROM oqmd_entries e
JOIN oqmd_formation_energies f ON f.entry_key = e.entry_key
WHERE f.gap_ev IS NOT NULL
ORDER BY f.gap_ev DESC, e.entry_key
LIMIT 5;
