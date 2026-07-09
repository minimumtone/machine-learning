-- hard: 元素比0.75を占める元素ごとの平均hull距離上位10
SELECT r.symbol,
       COUNT(*) AS n,
       ROUND(AVG(f.hull_distance)::numeric, 4) AS avg_hull_distance
FROM oqmd_element_ratios r
JOIN oqmd_formation_energies f ON f.entry_key = r.entry_key
WHERE r.atomic_ratio = 0.75
GROUP BY r.symbol
ORDER BY avg_hull_distance ASC, r.symbol ASC
LIMIT 10;
