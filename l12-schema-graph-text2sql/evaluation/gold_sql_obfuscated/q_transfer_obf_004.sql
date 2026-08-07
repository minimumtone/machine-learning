-- easy: 格子定数4Å未満
SELECT col_rhea, col_quebec, col_falcon
FROM tbl_zulu
WHERE col_falcon < 4.0
LIMIT 10000;
