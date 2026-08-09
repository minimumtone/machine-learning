-- easy: Niを含むエントリ
SELECT DISTINCT e.col_rhea, e.col_quebec
FROM tbl_zulu e
JOIN tbl_juliet r ON r.col_rhea = e.col_rhea
WHERE r.col_papa = 'Ni'
LIMIT 10000;
