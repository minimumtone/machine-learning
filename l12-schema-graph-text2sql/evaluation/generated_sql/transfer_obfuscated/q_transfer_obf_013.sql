SELECT z.col_quebec
FROM tbl_zulu z
WHERE EXISTS (
    SELECT 1
    FROM tbl_juliet j_ni
    WHERE j_ni.col_rhea = z.col_rhea
      AND j_ni.col_papa = 'Ni'
)
AND EXISTS (
    SELECT 1
    FROM tbl_juliet j_al
    WHERE j_al.col_rhea = z.col_rhea
      AND j_al.col_papa = 'Al'
)
ORDER BY z.col_quebec
LIMIT 10000;
