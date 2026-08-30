SELECT m.formula
FROM material_entry m
WHERE m.number_of_elements >= 3
ORDER BY m.formula
LIMIT 10000;
