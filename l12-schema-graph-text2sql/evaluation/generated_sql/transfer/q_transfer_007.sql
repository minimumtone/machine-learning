SELECT COUNT(DISTINCT entry_key) AS on_hull_compound_count
FROM oqmd_formation_energies
WHERE on_hull = TRUE
LIMIT 10000;
