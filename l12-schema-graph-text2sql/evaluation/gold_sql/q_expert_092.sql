SELECT s.prototype, AVG(et.bulk_modulus_vrh) AS avg_bulk, COUNT(*) AS cnt
FROM structure s
JOIN calculation cal_et ON cal_et.entry_id = s.entry_id
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
GROUP BY s.prototype
ORDER BY avg_bulk DESC
LIMIT 10000;
