SELECT
  col_papa AS element,
  COUNT(DISTINCT col_rhea) AS entry_count
FROM tbl_juliet
GROUP BY col_papa
ORDER BY entry_count DESC
LIMIT 10;
