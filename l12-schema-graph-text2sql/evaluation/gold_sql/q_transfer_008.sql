-- medium: 形成エネルギー -0.5未満
SELECT e.composition_formula, f.delta_e
FROM oqmd_entries e
JOIN oqmd_formation_energies f ON f.entry_key = e.entry_key
WHERE f.delta_e < -0.5
ORDER BY f.delta_e ASC
LIMIT 10000;
