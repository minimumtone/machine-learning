SELECT m.formula, dos.spin_polarized
FROM material_entry m
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN density_of_states dos ON dos.calculation_id = calc.calculation_id
WHERE dos.spin_polarized = TRUE
ORDER BY m.formula
LIMIT 10000;
