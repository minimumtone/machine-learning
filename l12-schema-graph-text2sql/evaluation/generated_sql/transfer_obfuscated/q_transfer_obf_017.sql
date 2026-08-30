SELECT
  tbl_juliet.col_papa,
  AVG(tbl_delta.col_xenon) AS avg_hull_distance
FROM tbl_juliet
JOIN tbl_zulu ON tbl_juliet.col_rhea = tbl_zulu.col_rhea
JOIN tbl_delta ON tbl_delta.col_rhea = tbl_zulu.col_rhea
WHERE tbl_juliet.col_juliet = 0.75
GROUP BY tbl_juliet.col_papa
ORDER BY AVG(tbl_delta.col_xenon) ASC
LIMIT 10;
