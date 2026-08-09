-- hard: A-siteがCoの化合物で形成エネルギーが低い5件
SELECT e.col_quebec, f.col_luna
FROM tbl_zulu e
JOIN tbl_juliet r ON r.col_rhea = e.col_rhea
JOIN tbl_delta f ON f.col_rhea = e.col_rhea
WHERE r.col_papa = 'Co' AND r.col_zulu = 'A-site'
ORDER BY f.col_luna ASC
LIMIT 5;
