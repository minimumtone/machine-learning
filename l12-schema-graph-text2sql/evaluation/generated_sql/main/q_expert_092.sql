SELECT cp.property_name, AVG(cp.value) AS average_bulk_modulus
FROM calculation calc
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE cp.property_name = 'bulk_modulus'
GROUP BY cp.property_name
ORDER BY average_bulk_modulus DESC
LIMIT 10000;
