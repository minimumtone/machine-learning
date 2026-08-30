-- very_hard: CTE A-site元素ごとの平均補正生成エンタルピー
WITH enthalpy AS (
    SELECT e.col_rhea, e.col_quebec, f.col_luna,
           SUM(r.col_juliet * rs.col_tango) AS weighted_ref
    FROM tbl_zulu e
    JOIN tbl_delta f ON f.col_rhea = e.col_rhea
    JOIN tbl_juliet r ON r.col_rhea = e.col_rhea
    JOIN tbl_victor rs ON rs.col_papa = r.col_papa
    WHERE e.col_delta = 'L12' AND f.col_hotel = true
    GROUP BY e.col_rhea, e.col_quebec, f.col_luna
)
SELECT ra.col_papa AS a_site,
       COUNT(*) AS n_compounds,
       ROUND(AVG(en.col_luna - en.weighted_ref)::numeric, 4) AS avg_enthalpy
FROM enthalpy en
JOIN tbl_juliet ra
    ON ra.col_rhea = en.col_rhea AND ra.col_zulu = 'A-site'
GROUP BY ra.col_papa
ORDER BY avg_enthalpy ASC;
