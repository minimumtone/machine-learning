-- very_hard: CTE 凸包上L12化合物の補正生成エンタルピー -0.3未満
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
SELECT col_quebec,
       ROUND((col_luna - weighted_ref)::numeric, 4) AS enthalpy_vs_element_ground_states
FROM enthalpy
WHERE col_luna - weighted_ref < -0.3
ORDER BY enthalpy_vs_element_ground_states ASC
LIMIT 20;
