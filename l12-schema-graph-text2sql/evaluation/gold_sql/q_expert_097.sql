SELECT method, COUNT(*) AS cnt
FROM calculation
GROUP BY method
ORDER BY cnt DESC
LIMIT 10000;
