-- medium: 元素ごとのエントリ数上位10
SELECT r.col_papa, COUNT(DISTINCT r.col_rhea) AS n_entries
FROM tbl_juliet r
GROUP BY r.col_papa
ORDER BY n_entries DESC, r.col_papa ASC
LIMIT 10;
