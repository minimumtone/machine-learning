-- medium: 結晶系ごとの平均形成エネルギー
SELECT e.col_halo,
       COUNT(*) AS n,
       ROUND(AVG(f.col_luna)::numeric, 4) AS avg_delta_e
FROM tbl_zulu e
JOIN tbl_delta f ON f.col_rhea = e.col_rhea
WHERE e.col_halo IS NOT NULL
GROUP BY e.col_halo
ORDER BY avg_delta_e ASC;
