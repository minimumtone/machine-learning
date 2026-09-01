SELECT m.formula, m.chemical_system
FROM material_entry m
ORDER BY m.chemical_system, m.formula
LIMIT 10000;
