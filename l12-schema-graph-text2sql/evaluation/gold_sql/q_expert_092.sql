SELECT s.prototype, AVG(et.bulk_modulus_vrh) AS avg_bulk, COUNT(*) AS cnt
FROM structure s
JOIN elastic_tensor et ON et.entry_id = s.entry_id
GROUP BY s.prototype
ORDER BY avg_bulk DESC
LIMIT 10000;
