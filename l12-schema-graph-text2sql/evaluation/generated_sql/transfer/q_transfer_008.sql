SELECT e.composition_formula, fe.delta_e
FROM oqmd_entries e
JOIN oqmd_formation_energies fe ON fe.entry_key = e.entry_key
WHERE fe.delta_e < -0.5
ORDER BY fe.delta_e ASC
LIMIT 10000;
