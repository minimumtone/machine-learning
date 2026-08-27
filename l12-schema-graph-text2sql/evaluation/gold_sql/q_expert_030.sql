SELECT prototype, COUNT(*) AS cnt
FROM structure
GROUP BY prototype
ORDER BY cnt DESC, prototype
LIMIT 10000;
