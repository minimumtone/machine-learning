SELECT tbl_zulu.col_quebec, tbl_delta.col_iris
FROM tbl_delta
JOIN tbl_zulu ON tbl_delta.col_rhea = tbl_zulu.col_rhea
WHERE tbl_delta.col_iris IS NOT NULL
ORDER BY tbl_delta.col_iris DESC
LIMIT 5;
