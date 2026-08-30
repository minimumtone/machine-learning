SELECT
  tbl_zulu.col_quebec,
  tbl_zulu.col_apollo,
  tbl_delta.col_iris
FROM tbl_zulu
JOIN tbl_delta ON tbl_delta.col_rhea = tbl_zulu.col_rhea
WHERE tbl_zulu.col_apollo = 221
  AND tbl_delta.col_iris > 0
ORDER BY tbl_delta.col_iris DESC
LIMIT 10000;
