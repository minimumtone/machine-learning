SELECT cp.property_name, AVG(cp.value) AS average_bulk_modulus
FROM calculated_property cp
WHERE cp.property_name = 'bulk_modulus'
GROUP BY cp.property_name
LIMIT 10000;
