SELECT col_quebec, col_delta
FROM tbl_zulu
WHERE col_delta IN ('L12', 'L1₂')
ORDER BY col_quebec
LIMIT 10000;
