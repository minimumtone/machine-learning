SELECT DISTINCT tbl_zulu.col_quebec
FROM tbl_zulu
JOIN tbl_juliet ON tbl_juliet.col_rhea = tbl_zulu.col_rhea
WHERE tbl_juliet.col_papa = 'Ni'
ORDER BY tbl_zulu.col_quebec
LIMIT 10000;
