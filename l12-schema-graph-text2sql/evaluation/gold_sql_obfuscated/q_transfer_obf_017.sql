-- hard: 元素比0.75を占める元素ごとの平均hull距離上位10
SELECT r.col_papa,
       COUNT(*) AS n,
       ROUND(AVG(f.col_xenon)::numeric, 4) AS avg_hull_distance
FROM tbl_juliet r
JOIN tbl_delta f ON f.col_rhea = r.col_rhea
WHERE r.col_juliet = 0.75
GROUP BY r.col_papa
ORDER BY avg_hull_distance ASC, r.col_papa ASC
LIMIT 10;
