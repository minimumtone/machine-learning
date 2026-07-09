-- medium: hull距離0.05未満の準安定化合物数
SELECT COUNT(*) AS n_near_hull
FROM oqmd_formation_energies
WHERE hull_distance < 0.05 AND on_hull = false;
