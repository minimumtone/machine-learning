SELECT calc.method, COUNT(DISTINCT calc.entry_id) AS entry_count
FROM calculation calc
GROUP BY calc.method
ORDER BY entry_count DESC
LIMIT 10000;
