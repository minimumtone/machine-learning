SELECT prototype, COUNT(*) AS cnt
FROM structure
GROUP BY prototype
ORDER BY cnt DESC
LIMIT 10000;
