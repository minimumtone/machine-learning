SELECT e.composition_formula, fe.gap_ev
FROM oqmd_entries e
JOIN oqmd_formation_energies fe ON fe.entry_key = e.entry_key
WHERE fe.gap_ev IS NOT NULL
ORDER BY fe.gap_ev DESC
LIMIT 5;
