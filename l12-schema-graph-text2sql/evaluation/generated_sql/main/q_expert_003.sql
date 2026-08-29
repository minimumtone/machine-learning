SELECT m.formula, m.number_of_elements
FROM material_entry m
WHERE m.number_of_elements >= 3
ORDER BY m.number_of_elements DESC, m.formula ASC
LIMIT 10000;
