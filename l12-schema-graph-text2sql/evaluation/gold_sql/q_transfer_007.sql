-- medium: 凸包上の化合物数
SELECT COUNT(*) AS n_on_hull
FROM oqmd_formation_energies
WHERE on_hull = true;
