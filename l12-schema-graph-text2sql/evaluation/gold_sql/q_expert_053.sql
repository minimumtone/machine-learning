SELECT m.entry_id, m.formula
FROM material_entry m
JOIN calculation cal_dos ON cal_dos.entry_id = m.entry_id AND cal_dos.calculation_type = 'relaxation'
JOIN density_of_states dos ON dos.calculation_id = cal_dos.calculation_id
WHERE dos.spin_polarized = TRUE
ORDER BY m.formula
LIMIT 10000;
