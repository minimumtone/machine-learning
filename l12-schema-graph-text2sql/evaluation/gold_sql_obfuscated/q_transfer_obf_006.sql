-- easy: バンドギャップ最大5件
SELECT e.col_quebec, f.col_iris
FROM tbl_zulu e
JOIN tbl_delta f ON f.col_rhea = e.col_rhea
WHERE f.col_iris IS NOT NULL
ORDER BY f.col_iris DESC
LIMIT 5;
