-- hard: 空間群221でバンドギャップ>0の化合物
SELECT e.col_quebec, f.col_iris
FROM tbl_zulu e
JOIN tbl_delta f ON f.col_rhea = e.col_rhea
WHERE e.col_apollo = 221 AND f.col_iris > 0
ORDER BY f.col_iris DESC
LIMIT 10000;
