SELECT COUNT(*) AS metastable_compound_count
FROM oqmd_formation_energies
WHERE on_hull = false
  AND hull_distance < 0.05
LIMIT 10000;
