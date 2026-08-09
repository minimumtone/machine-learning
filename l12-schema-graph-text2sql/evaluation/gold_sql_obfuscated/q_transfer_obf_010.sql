-- medium: Alを含む安定(凸包上)化合物
SELECT DISTINCT e.col_quebec, f.col_luna
FROM tbl_zulu e
JOIN tbl_delta f ON f.col_rhea = e.col_rhea
JOIN tbl_juliet r ON r.col_rhea = e.col_rhea
WHERE r.col_papa = 'Al' AND f.col_hotel = true
ORDER BY f.col_luna ASC
LIMIT 10000;
