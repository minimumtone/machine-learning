-- medium: 元素ごとのエントリ数上位10
SELECT r.symbol, COUNT(DISTINCT r.entry_key) AS n_entries
FROM oqmd_element_ratios r
GROUP BY r.symbol
ORDER BY n_entries DESC, r.symbol ASC
LIMIT 10;
