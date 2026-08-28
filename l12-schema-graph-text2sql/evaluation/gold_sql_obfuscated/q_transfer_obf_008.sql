-- medium: 形成エネルギー -0.5未満
SELECT e.col_quebec, f.col_luna
FROM tbl_zulu e
JOIN tbl_delta f ON f.col_rhea = e.col_rhea
WHERE f.col_luna < -0.5
ORDER BY f.col_luna ASC, e.col_rhea ASC
LIMIT 10000;
