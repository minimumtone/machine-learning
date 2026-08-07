-- very_hard: CTE 生成エンタルピー計算（Co3Ti相当）
WITH target AS (
    SELECT e.col_rhea, e.col_quebec, f.col_luna
    FROM tbl_zulu e
    JOIN tbl_delta f ON f.col_rhea = e.col_rhea
    WHERE e.col_quebec = 'Co3Ti'
    LIMIT 1
),
ref AS (
    SELECT t.col_rhea,
           SUM(r.col_juliet * rs.col_tango) AS weighted_ref
    FROM target t
    JOIN tbl_juliet r ON r.col_rhea = t.col_rhea
    JOIN tbl_victor rs ON rs.col_papa = r.col_papa
    GROUP BY t.col_rhea
)
SELECT t.col_quebec, t.col_luna, ref.weighted_ref,
       t.col_luna - ref.weighted_ref AS corrected_enthalpy
FROM target t
JOIN ref ON ref.col_rhea = t.col_rhea;
