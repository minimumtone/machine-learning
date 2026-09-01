SELECT calc.method, COUNT(DISTINCT m.entry_id) AS entry_count
FROM calculation calc
JOIN material_entry m ON m.entry_id = calc.entry_id
GROUP BY calc.method
ORDER BY entry_count DESC
LIMIT 10000;
