SELECT c.functional, AVG(et.bulk_modulus_vrh) AS avg_bulk_modulus FROM elastic_tensor et JOIN calculation c ON c.calculation_id = et.calculation_id GROUP BY c.functional ORDER BY c.functional;
