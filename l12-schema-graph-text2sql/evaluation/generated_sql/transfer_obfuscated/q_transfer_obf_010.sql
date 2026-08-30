SELECT
  tbl_zulu.col_quebec,
  tbl_delta.col_luna
FROM tbl_zulu
JOIN tbl_delta
  ON tbl_delta.col_rhea = tbl_zulu.col_rhea
WHERE tbl_delta.col_hotel = TRUE
  AND EXISTS (
    SELECT 1
    FROM tbl_juliet
    WHERE tbl_juliet.col_rhea = tbl_zulu.col_rhea
      AND tbl_juliet.col_papa = 'Al'
  )
ORDER BY tbl_delta.col_luna ASC
LIMIT 10000;
