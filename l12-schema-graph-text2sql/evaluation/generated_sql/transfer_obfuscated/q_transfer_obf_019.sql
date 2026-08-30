WITH corrected AS (
    SELECT
        z.col_quebec,
        d.col_luna - SUM(j.col_juliet * v.col_tango) AS corrected_formation_energy
    FROM tbl_zulu z
    JOIN tbl_delta d ON d.col_rhea = z.col_rhea
    JOIN tbl_juliet j ON j.col_rhea = z.col_rhea
    JOIN tbl_xray x ON j.col_papa = x.col_papa
    JOIN tbl_victor v ON v.col_papa = x.col_papa
    WHERE d.col_hotel = TRUE
      AND z.col_delta = 'L12'
    GROUP BY z.col_rhea, z.col_quebec, d.col_luna
)
SELECT
    col_quebec,
    ROUND(corrected_formation_energy::numeric, 4) AS corrected_formation_energy
FROM corrected
WHERE corrected_formation_energy < -0.3
ORDER BY corrected_formation_energy ASC
LIMIT 10000;
