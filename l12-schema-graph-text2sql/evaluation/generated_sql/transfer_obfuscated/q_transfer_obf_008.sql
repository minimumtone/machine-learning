SELECT tbl_zulu.col_quebec, tbl_delta.col_luna
FROM tbl_delta
JOIN tbl_zulu ON tbl_delta.col_rhea = tbl_zulu.col_rhea
WHERE tbl_delta.col_luna < -0.5
ORDER BY tbl_delta.col_luna ASC
LIMIT 10000;
