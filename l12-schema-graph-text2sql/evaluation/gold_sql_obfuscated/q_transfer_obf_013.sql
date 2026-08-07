-- hard: Ni-Al両方を含む化合物
SELECT DISTINCT e.col_rhea, e.col_quebec
FROM tbl_zulu e
JOIN tbl_juliet ra ON ra.col_rhea = e.col_rhea AND ra.col_papa = 'Ni'
JOIN tbl_juliet rb ON rb.col_rhea = e.col_rhea AND rb.col_papa = 'Al'
LIMIT 10000;
